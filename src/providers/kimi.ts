/**
 * Kimi Code (kimi.com / kimi.ai membership) subscription provider: the OAuth
 * device-authorization flow Kimi Code CLI uses against `auth.<site>`, a
 * standard refresh-token grant, and streaming against the OpenAI-compatible
 * chat completions endpoint at `api.<site>/coding/v1`. Each stored session
 * remembers its site, so kimi.com and kimi.ai accounts can coexist.
 *
 * Protocol facts mirror MoonshotAI/kimi-cli (`kimi_cli/auth/oauth.py`,
 * `kimi_cli/auth/platforms.py`, `kosong/chat_provider/kimi.py`):
 * - every auth and API request carries the CLI's `X-Msh-*` device identity
 *   headers and a `KimiCLI/<version>` user agent;
 * - thinking is toggled with `thinking: { type: "enabled" | "disabled" }`,
 *   and assistant reasoning is replayed as `reasoning_content`;
 * - `/models` lists `context_length`, `supports_reasoning`,
 *   `supports_image_in`, `display_name`; `/usages` reports quota windows.
 */

import { randomUUID } from 'node:crypto'
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { arch, hostname, release, type, version as osVersion } from 'node:os'
import { dirname } from 'node:path'
import { EMPTY_RESPONSE_CODE, errorChain, LlmAdapter, LlmError, ReasoningEffortId } from '@deepseek-ai/dsh-llm'
import type {
  GenerateOptions,
  LlmModelInfo,
  LlmProviderInfo,
  LlmResolvedModelInfo,
  StreamChunk,
} from '@deepseek-ai/dsh-llm'
import type { AttachmentStore } from '@deepseek-ai/dsh-attachment'
import { dshHomePath } from '@deepseek-ai/dsh-home-paths'
import type { DeviceFlowSpec, DeviceTokenResult } from '../auth/device-flow.js'
import { decodeJwtPayload } from '../auth/jwt.js'
import type { KimiSession, ProviderId } from '../auth/store.js'
import { proxiedFetch } from '../http.js'
import { resolveImages } from '../translate/resolved.js'
import { streamChatCompletions, toChatMessages, toChatTools } from '../translate/chat-completions.js'
import type { PoolAdapter } from './pool.js'
import {
  discoverAcrossAccounts,
  discoverOrRetryAuth,
  httpLlmError,
  idleWatchdog,
  isDiscoveryAborted,
  isMissingOrInvalidCredential,
  mapFetchFailure,
  mergeReasoning,
  ModelCatalogCache,
  OAuthEndpointError,
  oauthEndpointError,
} from './common.js'
import type { CatalogPersistence, DiscoveredModel, FetchFn, ModelEntry, ProviderUsage, UsageWindow } from './common.js'
import { AccountTokenManager, DISCOVERY_TIMEOUT_MS, unionAccountCatalogs } from './accounts.js'
import { DEFAULT_RATE_LIMIT_WAIT, DEFAULT_RETRY, subscriptionRetryPolicy } from './rate-limit.js'
import type { RateLimitWait } from './rate-limit.js'

/** Public OAuth client id of Kimi Code CLI (accepted by both sites). */
export const KIMI_CLIENT_ID = '17e5f671-d194-4dfb-9706-5516cb48c098'

/**
 * The Kimi site an account belongs to. kimi.com and kimi.ai run separate
 * auth servers and API gateways (`auth.<site>`, `api.<site>/coding/v1`); a
 * session must refresh and call the API on the site that issued it.
 */
export type KimiSite = 'kimi.com' | 'kimi.ai'
export const KIMI_SITES: readonly KimiSite[] = ['kimi.com', 'kimi.ai']
export const DEFAULT_KIMI_SITE: KimiSite = 'kimi.com'

/** The endpoints of one Kimi site. */
export interface KimiEndpoints {
  deviceCodeUrl: string
  tokenUrl: string
  chatUrl: string
  modelsUrl: string
  usagesUrl: string
}

/**
 * Resolve one site's endpoints.
 * @param site - the site; sessions stored without one are kimi.com.
 * @returns the auth and API URLs for that site.
 */
export function kimiEndpoints(site: KimiSite | undefined = DEFAULT_KIMI_SITE): KimiEndpoints {
  const host = site === 'kimi.ai' ? 'kimi.ai' : 'kimi.com'
  const auth = `https://auth.${host}/api/oauth`
  const api = `https://api.${host}/coding/v1`
  return {
    deviceCodeUrl: `${auth}/device_authorization`,
    tokenUrl: `${auth}/token`,
    chatUrl: `${api}/chat/completions`,
    modelsUrl: `${api}/models`,
    usagesUrl: `${api}/usages`,
  }
}

const KIMI_COM = kimiEndpoints('kimi.com')
export const KIMI_DEVICE_CODE_URL = KIMI_COM.deviceCodeUrl
export const KIMI_TOKEN_URL = KIMI_COM.tokenUrl
export const KIMI_CHAT_URL = KIMI_COM.chatUrl
export const KIMI_MODELS_URL = KIMI_COM.modelsUrl
export const KIMI_USAGES_URL = KIMI_COM.usagesUrl
/** Kimi Code CLI version presented in the user agent and `X-Msh-Version`. */
export const KIMI_CLI_VERSION = '1.52.0'
/** Refresh when the access token has less than this much life left. */
export const KIMI_PREEMPT_MS = 5 * 60_000
const KIMI_CONTEXT_WINDOW = 262_144
const KIMI_DEFAULT_MAX_TOKENS = 32_000

/** Effort ids the picker offers for Kimi's binary thinking switch. */
const EFFORT_OFF = ReasoningEffortId('none')
const EFFORT_ON = ReasoningEffortId('high')

let cachedDeviceId: string | undefined

/**
 * Stable per-installation device id sent as `X-Msh-Device-Id` (Kimi binds
 * the OAuth grant to it). Persisted beside the auth store; an unwritable home
 * falls back to a per-process id rather than failing the login.
 */
export function kimiDeviceId(): string {
  if (cachedDeviceId !== undefined) return cachedDeviceId
  const path = dshHomePath('plugins', 'subscriptions', 'kimi-device-id')
  try {
    const stored = readFileSync(path, 'utf8').trim()
    if (/^[0-9a-f]{32}$/.test(stored)) return (cachedDeviceId = stored)
  } catch {
    // Missing file: mint one below.
  }
  const minted = randomUUID().replaceAll('-', '')
  try {
    mkdirSync(dirname(path), { recursive: true })
    writeFileSync(path, minted, { mode: 0o600 })
  } catch {
    // Best effort: an in-memory id still works for this process.
  }
  return (cachedDeviceId = minted)
}

/** Header values must be ASCII; strip anything else (hostnames may be localized). */
function ascii(value: string, fallback = 'unknown'): string {
  const cleaned = value.replace(/[^\x20-\x7e]/g, '').trim()
  return cleaned.length > 0 ? cleaned : fallback
}

function deviceModel(): string {
  const os = type()
  const name = os === 'Darwin' ? 'macOS' : os === 'Windows_NT' ? 'Windows' : os
  return `${name} ${release()} ${arch()}`.trim()
}

/**
 * The Kimi Code CLI identity headers sent on every auth and API request.
 * @returns headers to merge into Kimi requests.
 */
export function kimiHeaders(): Record<string, string> {
  return {
    'user-agent': `KimiCLI/${KIMI_CLI_VERSION}`,
    'x-msh-platform': 'kimi_cli',
    'x-msh-version': KIMI_CLI_VERSION,
    'x-msh-device-name': ascii(hostname()),
    'x-msh-device-model': ascii(deviceModel()),
    'x-msh-os-version': ascii(osVersion()),
    'x-msh-device-id': kimiDeviceId(),
  }
}

/**
 * The device-flow facts for the auth controller's DeviceFlowManager.
 * @param site - the Kimi site to authorize against.
 * @returns the flow spec for one attempt.
 */
export function kimiDeviceFlow(site: KimiSite = DEFAULT_KIMI_SITE): DeviceFlowSpec {
  const endpoints = kimiEndpoints(site)
  return {
    clientId: KIMI_CLIENT_ID,
    deviceCodeUrl: endpoints.deviceCodeUrl,
    tokenUrl: endpoints.tokenUrl,
    headers: kimiHeaders(),
    displayName: site,
  }
}

/** Best-effort display identity from the access token's claims. */
function kimiAccountOf(accessToken: string): string | undefined {
  const claims = decodeJwtPayload(accessToken)
  if (claims === undefined) return undefined
  for (const key of ['email', 'phone', 'name', 'preferred_username', 'user_id', 'sub']) {
    const value = claims[key]
    if (typeof value === 'string' && value.trim().length > 0) return value.trim()
  }
  return undefined
}

/** Expiry from `expires_in`, falling back to the JWT `exp` claim, then 15 minutes. */
function expiryOf(accessToken: string, expiresIn: number | undefined): number {
  if (expiresIn !== undefined && Number.isFinite(expiresIn) && expiresIn > 0) return Date.now() + expiresIn * 1000
  const exp = decodeJwtPayload(accessToken)?.exp
  if (typeof exp === 'number' && exp > 0) return exp * 1000
  return Date.now() + 15 * 60_000
}

/**
 * Turn the device flow's token response into a stored session.
 * @param tokens - the token response the device flow released.
 * @param site - the site that issued the tokens.
 * @returns the session to store.
 * @throws when Kimi issued no refresh token (the session could not be renewed).
 */
export function completeKimiLogin(tokens: DeviceTokenResult, site: KimiSite = DEFAULT_KIMI_SITE): KimiSession {
  if (tokens.refreshToken === undefined) {
    throw new Error('kimi token endpoint returned no refresh_token')
  }
  const account = kimiAccountOf(tokens.accessToken)
  return {
    accessToken: tokens.accessToken,
    refreshToken: tokens.refreshToken,
    expiresAt: expiryOf(tokens.accessToken, tokens.expiresIn),
    ...tokens.scope === undefined ? {} : { scope: tokens.scope },
    ...account === undefined ? {} : { account },
    // kimi.com stays implicit so sessions keep their pre-site shape.
    ...site === 'kimi.ai' ? { site } : {},
  }
}

/**
 * Refresh a Kimi session with the refresh-token grant. Kimi rotates the
 * refresh token; the previous one is kept only when the response omits it.
 * @param session - the stored session.
 * @param fetchFn - fetch implementation (injectable for tests).
 * @returns the fresh session to store.
 */
export async function refreshKimi(session: KimiSession, fetchFn: FetchFn = proxiedFetch): Promise<KimiSession> {
  const response = await fetchFn(kimiEndpoints(session.site).tokenUrl, {
    method: 'POST',
    headers: {
      ...kimiHeaders(),
      'accept': 'application/json',
      'content-type': 'application/x-www-form-urlencoded',
    },
    body: new URLSearchParams({
      client_id: KIMI_CLIENT_ID,
      grant_type: 'refresh_token',
      refresh_token: session.refreshToken,
    }).toString(),
  })
  if (!response.ok) throw await oauthEndpointError(response, 'kimi')
  const wire = await response.json() as {
    access_token?: string
    refresh_token?: string
    expires_in?: number | string
    scope?: string
  }
  if (typeof wire.access_token !== 'string' || wire.access_token.length === 0) {
    throw new Error('kimi token endpoint returned no access_token')
  }
  const account = kimiAccountOf(wire.access_token) ?? session.account
  const scope = typeof wire.scope === 'string' && wire.scope.length > 0 ? wire.scope : session.scope
  return {
    accessToken: wire.access_token,
    refreshToken: typeof wire.refresh_token === 'string' && wire.refresh_token.length > 0
      ? wire.refresh_token
      : session.refreshToken,
    expiresAt: expiryOf(wire.access_token, wire.expires_in === undefined ? undefined : Number(wire.expires_in)),
    ...scope === undefined ? {} : { scope },
    ...account === undefined ? {} : { account },
    ...session.site === undefined ? {} : { site: session.site },
  }
}

/**
 * Whether a Kimi refresh failure means the login is permanently gone.
 * @param error - the thrown refresh error.
 * @returns true when re-login is the only fix.
 */
export function isKimiPermanentRefreshError(error: unknown): boolean {
  return error instanceof OAuthEndpointError
    && (error.status === 401 || error.status === 403 || error.oauthCode === 'invalid_grant')
}

/** The /models entry subset this plugin reads. */
interface KimiWireModel {
  id?: string
  display_name?: string | null
  context_length?: number
  supports_reasoning?: boolean
  supports_image_in?: boolean
}

/**
 * Reasoning block for one model: Kimi exposes a binary thinking switch, so
 * it maps onto two efforts (`none` = thinking off, `high` = on). Models
 * whose id names thinking always reason and offer only `high`.
 */
function kimiReasoning(id: string, supportsReasoning: boolean): DiscoveredModel['reasoning'] | undefined {
  const lower = id.toLowerCase()
  if (lower.includes('thinking')) {
    return { efforts: [{ id: EFFORT_ON, name: 'On' }], defaultEffort: EFFORT_ON }
  }
  if (!supportsReasoning && !lower.startsWith('kimi-k2') && !lower.startsWith('kimi-for-coding')) return undefined
  return {
    efforts: [{ id: EFFORT_OFF, name: 'Off' }, { id: EFFORT_ON, name: 'On' }],
    defaultEffort: EFFORT_ON,
  }
}

/**
 * Fetch the live Kimi Code model list.
 * @param session - the stored session (used as-is; never refreshed here).
 * @param fetchFn - fetch implementation (injectable for tests).
 * @param signal - caller cancellation.
 * @returns discovered models in endpoint order.
 */
export async function fetchKimiModels(
  session: KimiSession,
  fetchFn: FetchFn = proxiedFetch,
  signal?: AbortSignal,
): Promise<DiscoveredModel[]> {
  const response = await fetchFn(kimiEndpoints(session.site).modelsUrl, {
    headers: {
      ...kimiHeaders(),
      'authorization': `Bearer ${session.accessToken}`,
      'accept': 'application/json',
    },
    ...signal === undefined ? {} : { signal },
  })
  if (!response.ok) throw await oauthEndpointError(response, 'kimi models')
  const payload = await response.json() as { data?: KimiWireModel[] }
  if (!Array.isArray(payload.data)) throw new Error('kimi models endpoint returned no data array')
  const seen = new Set<string>()
  const discovered: DiscoveredModel[] = []
  for (const entry of payload.data) {
    if (typeof entry.id !== 'string' || entry.id.length === 0 || seen.has(entry.id)) continue
    seen.add(entry.id)
    const vision = entry.supports_image_in === true || entry.id.toLowerCase().startsWith('kimi-k2')
    const reasoning = kimiReasoning(entry.id, entry.supports_reasoning === true)
    discovered.push({
      id: entry.id,
      name: typeof entry.display_name === 'string' && entry.display_name.length > 0 ? entry.display_name : entry.id,
      ...typeof entry.context_length === 'number' && Number.isInteger(entry.context_length) && entry.context_length > 0
        ? { contextWindow: entry.context_length }
        : {},
      inputModalities: vision ? ['text', 'image'] : ['text'],
      ...reasoning === undefined ? {} : { reasoning },
    })
  }
  if (discovered.length === 0) throw new Error('kimi models endpoint returned an empty catalog')
  return discovered
}

function toNumber(value: unknown): number | undefined {
  const parsed = typeof value === 'string' ? Number(value) : value
  return typeof parsed === 'number' && Number.isFinite(parsed) ? parsed : undefined
}

function resetOf(data: Record<string, unknown>): number | undefined {
  for (const key of ['reset_at', 'resetAt', 'reset_time', 'resetTime']) {
    const value = data[key]
    if (typeof value === 'string' && value.length > 0) {
      const parsed = Date.parse(value)
      if (Number.isFinite(parsed)) return parsed
    }
    const numeric = toNumber(value)
    if (numeric !== undefined && numeric > 0) return numeric < 1e12 ? numeric * 1000 : numeric
  }
  for (const key of ['reset_in', 'resetIn', 'ttl']) {
    const seconds = toNumber(data[key])
    if (seconds !== undefined && seconds > 0) return Date.now() + seconds * 1000
  }
  return undefined
}

function usedPercentOf(data: Record<string, unknown>): number | undefined {
  const limit = toNumber(data.limit)
  let used = toNumber(data.used)
  if (used === undefined) {
    const remaining = toNumber(data.remaining)
    if (remaining !== undefined && limit !== undefined) used = limit - remaining
  }
  if (used === undefined || limit === undefined || limit <= 0) return undefined
  return Math.min(100, Math.max(0, (used / limit) * 100))
}

function windowMinutes(item: Record<string, unknown>, detail: Record<string, unknown>): number | undefined {
  const window = typeof item.window === 'object' && item.window !== null ? item.window as Record<string, unknown> : {}
  const duration = toNumber(window.duration ?? item.duration ?? detail.duration)
  if (duration === undefined || duration <= 0) return undefined
  const unit = String(window.timeUnit ?? item.timeUnit ?? detail.timeUnit ?? '')
  if (unit.includes('MINUTE')) return duration
  if (unit.includes('HOUR')) return duration * 60
  if (unit.includes('DAY')) return duration * 1440
  return duration / 60
}

/**
 * Parse a Kimi `/usages` payload into usage windows: the `usage` summary is
 * the weekly quota; each `limits[]` entry is a rolling window (the 5-hour
 * one shows as `session`).
 * @param payload - the parsed JSON body.
 * @returns the plugin's usage shape.
 */
export function parseKimiUsage(payload: unknown): ProviderUsage {
  const windows: UsageWindow[] = []
  const root = typeof payload === 'object' && payload !== null ? payload as Record<string, unknown> : {}
  const limits = Array.isArray(root.limits) ? root.limits : []
  for (const raw of limits) {
    if (typeof raw !== 'object' || raw === null) continue
    const item = raw as Record<string, unknown>
    const detail = typeof item.detail === 'object' && item.detail !== null ? item.detail as Record<string, unknown> : item
    const usedPercent = usedPercentOf(detail)
    if (usedPercent === undefined) continue
    const minutes = windowMinutes(item, detail)
    const resetsAt = resetOf(detail)
    const label = [item.name, item.title, detail.name, detail.title].find(value => typeof value === 'string' && value.length > 0)
    windows.push({
      kind: minutes !== undefined && minutes <= 24 * 60 ? 'session' : minutes !== undefined && minutes >= 7 * 1440 ? 'weekly' : 'other',
      usedPercent,
      ...typeof label === 'string' ? { scope: label } : {},
      ...resetsAt === undefined ? {} : { resetsAt },
    })
  }
  if (typeof root.usage === 'object' && root.usage !== null) {
    const summary = root.usage as Record<string, unknown>
    const usedPercent = usedPercentOf(summary)
    if (usedPercent !== undefined) {
      const resetsAt = resetOf(summary)
      windows.push({ kind: 'weekly', usedPercent, ...resetsAt === undefined ? {} : { resetsAt } })
    }
  }
  return { supported: true, windows }
}

/**
 * Fetch one account's Kimi Code quota.
 * @param session - a fresh session.
 * @param fetchFn - fetch implementation (injectable for tests).
 * @param signal - caller cancellation.
 * @returns the account's usage windows.
 */
export async function fetchKimiUsage(
  session: KimiSession,
  fetchFn: FetchFn = proxiedFetch,
  signal?: AbortSignal,
): Promise<ProviderUsage> {
  const response = await fetchFn(kimiEndpoints(session.site).usagesUrl, {
    headers: {
      ...kimiHeaders(),
      'authorization': `Bearer ${session.accessToken}`,
      'accept': 'application/json',
    },
    ...signal === undefined ? {} : { signal },
  })
  if (!response.ok) throw await oauthEndpointError(response, 'kimi usages')
  return parseKimiUsage(await response.json())
}

/**
 * The chat completions request body for one generation.
 * @param options - the harness generate options.
 * @param messages - translated wire messages (images pre-resolved).
 * @param thinking - whether the model reasons for this request; undefined leaves the server default.
 * @returns the JSON body.
 */
export function kimiChatRequestBody(
  options: GenerateOptions,
  messages: Record<string, unknown>[],
  thinking: boolean | undefined,
): Record<string, unknown> {
  return {
    model: options.model,
    messages,
    ...options.tools !== undefined && options.tools.length > 0
      ? { tools: toChatTools(options.tools), tool_choice: 'auto' }
      : {},
    ...options.maxTokens !== undefined ? { max_completion_tokens: options.maxTokens } : {},
    ...thinking === undefined ? {} : { thinking: { type: thinking ? 'enabled' : 'disabled' } },
    ...options.sessionId !== undefined ? { prompt_cache_key: String(options.sessionId) } : {},
    stream: true,
    stream_options: { include_usage: true },
  }
}

/** Constructor dependencies for {@link KimiAdapter}. */
export interface KimiAdapterOptions {
  models: readonly ModelEntry[]
  streamIdleTimeoutMs: number
  tokens: AccountTokenManager<KimiSession>
  pool?: () => PoolAdapter | undefined
  discovery: boolean
  onWarn?: (message: string) => void
  fetchFn?: FetchFn
  resolveAttachments?: () => AttachmentStore | undefined
  catalogStore?: CatalogPersistence
  onCatalogChanged?: () => void
  accountCatalogStore?: (account: string) => CatalogPersistence
  rateLimit?: RateLimitWait
  defaultEffortOf?: (model: string) => string | undefined
}

/** Kimi Code wire adapter: one instance serves the `kimi` provider route. */
export class KimiAdapter extends LlmAdapter {
  private readonly catalog: ModelCatalogCache
  private readonly accountCatalogs = new Map<string, ModelCatalogCache>()
  private accountCatalogsInvalidated = false
  private readonly invalidatedAccounts = new Set<string>()
  private catalogOwner: string | undefined

  constructor(private readonly options: KimiAdapterOptions) {
    super()
    this.catalog = new ModelCatalogCache(options.catalogStore, undefined, options.onCatalogChanged)
  }

  private async fetchCatalog(account?: string, signal?: AbortSignal): Promise<DiscoveredModel[]> {
    return fetchKimiModels(await this.options.tokens.session(account), this.options.fetchFn, signal)
  }

  /** Drop cached catalogs after login/logout so the next list does not reuse a stale plan. */
  clearAccountCatalog(account?: string): void {
    if (account === undefined) {
      this.accountCatalogsInvalidated = true
      this.invalidatedAccounts.clear()
      for (const cache of this.accountCatalogs.values()) cache.invalidate()
      this.accountCatalogs.clear()
    } else {
      this.invalidatedAccounts.add(account)
      this.accountCatalogs.get(account)?.invalidate()
      this.accountCatalogs.delete(account)
    }
    if (account === undefined || this.catalogOwner === account || this.catalogOwner === undefined) {
      this.catalogOwner = undefined
      this.catalog.invalidate()
    }
  }

  private async catalogFor(account?: string): Promise<ModelCatalogCache> {
    const defaultKey = await this.options.tokens.defaultAccount()
    const key = account ?? defaultKey
    if (key === undefined || key === defaultKey) {
      if (this.catalogOwner !== undefined && this.catalogOwner !== defaultKey) this.catalog.invalidate()
      this.catalogOwner = defaultKey
      return this.catalog
    }
    let cache = this.accountCatalogs.get(key)
    if (cache === undefined) {
      cache = new ModelCatalogCache(this.options.accountCatalogStore?.(key), undefined, this.options.onCatalogChanged)
      const invalidated = this.invalidatedAccounts.delete(key)
      if (this.accountCatalogsInvalidated || invalidated) cache.invalidate()
      this.accountCatalogs.set(key, cache)
    }
    return cache
  }

  override providerInfo(provider: string): LlmProviderInfo {
    return { id: provider, name: 'Kimi Code' }
  }

  override providerRetryPolicy(provider: string) {
    return subscriptionRetryPolicy(
      DEFAULT_RETRY,
      this.options.rateLimit ?? DEFAULT_RATE_LIMIT_WAIT,
      `kimi: provider "${provider}" retryPolicy`,
    )
  }

  private staticModels(provider: string): LlmModelInfo[] {
    return this.options.models.map(model => ({
      provider,
      id: model.id,
      name: model.name ?? model.id,
      inputModalities: model.inputModalities ?? ['text'],
    }))
  }

  override async listModels(provider: string): Promise<readonly LlmModelInfo[]> {
    const own = await this.listOwnModels(provider, undefined, undefined, true)
    const pool = this.options.pool?.()
    if (pool === undefined) return own
    const extra = await pool.modelsForProvider(provider as ProviderId)
    const seen = new Set(own.map(model => model.id))
    return [...own, ...extra.filter(model => !seen.has(model.id))]
  }

  /** The provider's own catalog: union of every account, or one account when named. */
  async listOwnModels(provider: string, account?: string, signal?: AbortSignal, preferCached = false): Promise<readonly LlmModelInfo[]> {
    if (account === undefined) {
      const accounts = (await this.options.tokens.list()).map(entry => entry.key)
      if (accounts.length === 0) return []
      return unionAccountCatalogs(
        accounts,
        (key, accountSignal) => this.listOwnModels(provider, key, accountSignal, preferCached),
        { timeoutMs: DISCOVERY_TIMEOUT_MS, ...signal === undefined ? {} : { signal } },
      )
    }
    if (!await this.options.tokens.hasSession(account)) return []
    if (!this.options.discovery) return this.staticModels(provider)
    const catalog = await this.catalogFor(account)
    try {
      const discovered = await catalog.get(() => discoverOrRetryAuth(
        force => this.options.tokens.session(account, force),
        catalog,
        () => this.fetchCatalog(account, signal),
      ), preferCached)
      return discovered.map(model => ({
        provider,
        id: model.id,
        name: model.name,
        ...model.inputModalities === undefined ? {} : { inputModalities: model.inputModalities },
      }))
    } catch (error: unknown) {
      if (isDiscoveryAborted(error, signal)) throw error
      if (isMissingOrInvalidCredential(error)) return []
      this.options.onWarn?.(`kimi model discovery failed; using the built-in catalog (${errorChain(error)})`)
      return this.staticModels(provider)
    }
  }

  private async discovered(model: string): Promise<DiscoveredModel | undefined> {
    if (!this.options.discovery) return undefined
    const accounts = (await this.options.tokens.list()).map(entry => entry.key)
    return discoverAcrossAccounts(accounts, async account => {
      const catalog = await this.catalogFor(account)
      const models = await catalog.resolve(() => this.fetchCatalog(account))
      return models?.find(entry => entry.id === model)
    })
  }

  /** The model's reasoning block: discovered, else inferred from the id (static catalogs). */
  private async reasoningFor(model: string): Promise<DiscoveredModel['reasoning'] | undefined> {
    const discovered = await this.discovered(model)
    return discovered?.reasoning ?? (discovered === undefined ? kimiReasoning(model, false) : undefined)
  }

  override async resolveModel(provider: string, model: string): Promise<LlmResolvedModelInfo> {
    const pool = this.options.pool?.()
    if (pool !== undefined && await pool.owns(provider as ProviderId, model)) {
      return pool.resolveModel(provider, model)
    }
    return this.resolveOwnModel(provider, model)
  }

  async resolveOwnModel(provider: string, model: string): Promise<LlmResolvedModelInfo> {
    const discovered = await this.discovered(model)
    const configured = this.options.models.find(entry => entry.id === model)
    const base = discovered?.reasoning ?? (discovered === undefined ? kimiReasoning(model, false) : undefined)
    const reasoning = mergeReasoning(this.options.defaultEffortOf?.(model), base)
    return {
      provider,
      id: model,
      name: discovered?.name ?? configured?.name ?? model,
      inputModalities: discovered?.inputModalities ?? configured?.inputModalities ?? ['text'],
      context: { contextWindow: discovered?.contextWindow ?? configured?.contextWindow ?? KIMI_CONTEXT_WINDOW },
      defaultMaxTokens: configured?.maxTokens ?? KIMI_DEFAULT_MAX_TOKENS,
      ...reasoning === undefined ? {} : { reasoning },
    }
  }

  async *stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    const pool = this.options.pool?.()
    if (pool !== undefined && await pool.owns(options.provider as ProviderId, options.model)) {
      yield* pool.stream(options)
      return
    }
    yield* this.streamCore(options)
  }

  /** Pool seam: stream through one specific account instead of the default. */
  streamAccount(options: GenerateOptions, account: string): AsyncIterable<StreamChunk> {
    return this.streamCore(options, account)
  }

  private async *streamCore(options: GenerateOptions, account?: string): AsyncIterable<StreamChunk> {
    const watchdog = idleWatchdog(options.signal, this.options.streamIdleTimeoutMs)
    try {
      const reasoning = await this.reasoningFor(options.model)
      // Thinking follows the selected effort; a model without a reasoning
      // block keeps the server default.
      const thinking = reasoning === undefined
        ? undefined
        : options.reasoningEffort === undefined
          ? (reasoning.defaultEffort ?? EFFORT_ON) !== EFFORT_OFF
          : options.reasoningEffort !== EFFORT_OFF
      let session = await this.options.tokens.session(account)
      let response = await this.request(options, session, watchdog.signal, thinking)
      if (response.status === 401) {
        session = await this.options.tokens.session(account, true)
        response = await this.request(options, session, watchdog.signal, thinking)
      }
      if (!response.ok) {
        throw await httpLlmError(response, 'kimi API', {
          ...this.options.onWarn === undefined ? {} : { onWarn: this.options.onWarn },
        })
      }
      if (response.body === null) throw new LlmError('kimi API returned no response body', EMPTY_RESPONSE_CODE)
      yield* streamChatCompletions(response.body, () => { watchdog.pulse() })
    } catch (error: unknown) {
      throw mapFetchFailure('kimi API', error, watchdog, options.signal)
    } finally {
      watchdog.stop()
    }
  }

  private async request(
    options: GenerateOptions,
    session: KimiSession,
    signal: AbortSignal,
    thinking: boolean | undefined,
  ): Promise<Response> {
    const messages = await resolveImages(options.messages, this.options.resolveAttachments?.(), signal)
    const body = kimiChatRequestBody(
      options,
      toChatMessages(messages, options.system, { replayReasoning: true }),
      thinking,
    )
    return (this.options.fetchFn ?? proxiedFetch)(kimiEndpoints(session.site).chatUrl, {
      method: 'POST',
      headers: {
        ...kimiHeaders(),
        'authorization': `Bearer ${session.accessToken}`,
        'accept': 'text/event-stream',
        'content-type': 'application/json',
      },
      body: JSON.stringify(body),
      signal,
    })
  }
}
