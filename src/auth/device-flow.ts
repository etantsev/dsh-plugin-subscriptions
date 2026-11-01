/**
 * OAuth device-authorization flow (RFC 8628) — GitHub Copilot and Kimi Code —
 * for providers that cannot use the loopback redirect engine: no redirect URI, no PKCE, no client
 * secret. The user opens a verification URL and types a short code while the
 * plugin polls the token endpoint until GitHub releases the access token.
 * The management model (one attempt per provider, `isBusy`/`pending`/`cancel`)
 * mirrors {@link OAuthFlowManager} so the auth controller can treat both
 * engines uniformly.
 */
import { proxiedFetch } from '../http.js'

/** Default poll interval when the device-code response omits one. */
const DEFAULT_INTERVAL_SEC = 5

/** Default device-code lifetime when the response omits one (GitHub: 15 minutes). */
const DEFAULT_EXPIRES_IN_SEC = 900

/** Static per-provider device-flow facts. */
export interface DeviceFlowSpec {
  /** OAuth App / GitHub App client id the device code is requested for. */
  clientId: string
  /** Scope string requested at device-code time; omitted from the request when absent. */
  scope?: string
  /** Device-code endpoint (e.g. `https://github.com/login/device/code`). */
  deviceCodeUrl: string
  /** Token polling endpoint (e.g. `https://github.com/login/oauth/access_token`). */
  tokenUrl: string
  /** Extra headers sent on the device-code and token-poll requests (e.g. Kimi's `X-Msh-*` device identity). */
  headers?: Record<string, string>
  /** Provider name used in user-facing messages (defaults to the provider route). */
  displayName?: string
  /** Fetch implementation (injectable for tests). */
  fetchFn?: typeof fetch
}

/** The full token response a device flow released. */
export interface DeviceTokenResult {
  accessToken: string
  /** OAuth refresh token, when the provider issues one (GitHub's OAuth App flow does not). */
  refreshToken?: string
  /** Access-token lifetime in seconds, when disclosed. */
  expiresIn?: number
  /** Granted scope string, when disclosed. */
  scope?: string
}

/** One in-flight device-flow login attempt. */
export interface DeviceAttempt {
  /** URL the user opens to authorize (e.g. `https://github.com/login/device`). */
  readonly verificationUrl: string
  /** Short code the user types at the verification URL. */
  readonly userCode: string
  /**
   * Poll until GitHub releases the access token.
   * @returns the GitHub OAuth access token; rejects on timeout, denial, or cancel.
   */
  waitToken(): Promise<string>
  /** Like {@link waitToken}, but resolves the whole token response (refresh token, lifetime). */
  waitTokens(): Promise<DeviceTokenResult>
  /** Abort the attempt; `waitToken` rejects with a cancellation error. */
  cancel(): void
}

/** Raw device-code response shape (subset). */
interface DeviceCodeWire {
  device_code?: string
  user_code?: string
  verification_uri?: string
  /** RFC 8628 §3.3.1: verification URL with the user code embedded. */
  verification_uri_complete?: string
  interval?: number
  expires_in?: number
}

/** Raw token-poll response shape (subset): success or an RFC 8628 error code. */
interface DeviceTokenWire {
  access_token?: string
  refresh_token?: string
  expires_in?: number | string
  scope?: string
  error?: string
  error_description?: string
}

function nonEmptyString(value: unknown): string | undefined {
  return typeof value === 'string' && value.length > 0 ? value : undefined
}

/** Sleep for `ms`, rejecting early when the signal aborts. */
function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(signal.reason instanceof Error ? signal.reason : new Error('aborted'))
      return
    }
    const timer = setTimeout(() => {
      signal.removeEventListener('abort', onAbort)
      resolve()
    }, ms)
    timer.unref()
    const onAbort = (): void => {
      clearTimeout(timer)
      reject(signal.reason instanceof Error ? signal.reason : new Error('aborted'))
    }
    signal.addEventListener('abort', onAbort, { once: true })
  })
}

/**
 * Own the set of in-flight device-flow attempts, keyed by provider. One
 * attempt per provider at a time; an attempt removes itself when it settles.
 */
export class DeviceFlowManager {
  private attempts = new Map<string, DeviceAttempt>()

  /**
   * Whether a device-flow attempt is running for one provider.
   * @param provider - the provider route.
   * @returns true while an attempt is polling.
   */
  isBusy(provider: string): boolean {
    return this.attempts.has(provider)
  }

  /**
   * The pending attempt for one provider, when any.
   * @param provider - the provider route.
   * @returns the in-flight attempt, or `undefined`.
   */
  pending(provider: string): DeviceAttempt | undefined {
    return this.attempts.get(provider)
  }

  /**
   * Start a device-flow attempt: request a device code, then poll the token
   * endpoint in the background of `waitToken`.
   * @param provider - the provider route (one attempt at a time).
   * @param spec - static flow facts for this provider.
   * @returns the live attempt; its `waitToken()` settles the login.
   * @throws when an attempt is already running or the device-code request fails.
   */
  async start(provider: string, spec: DeviceFlowSpec): Promise<DeviceAttempt> {
    if (this.attempts.has(provider)) {
      throw new Error(`a ${provider} login attempt is already in progress`)
    }
    const fetchFn = spec.fetchFn ?? proxiedFetch
    const label = spec.displayName ?? provider
    const headers = {
      ...spec.headers,
      'accept': 'application/json',
      'content-type': 'application/x-www-form-urlencoded',
    }
    const response = await fetchFn(spec.deviceCodeUrl, {
      method: 'POST',
      headers,
      body: new URLSearchParams({
        client_id: spec.clientId,
        ...spec.scope === undefined ? {} : { scope: spec.scope },
      }).toString(),
    })
    if (!response.ok) {
      throw new Error(`${provider} device-code request failed (HTTP ${String(response.status)})`)
    }
    const wire = await response.json() as DeviceCodeWire
    // Prefer the plain URL (the UI shows the code beside it); providers that
    // only send the complete form (Kimi may leave verification_uri empty)
    // still work, since that URL carries the code itself.
    const verificationUrl = nonEmptyString(wire.verification_uri) ?? nonEmptyString(wire.verification_uri_complete)
    if (typeof wire.device_code !== 'string' || wire.device_code.length === 0
      || typeof wire.user_code !== 'string' || wire.user_code.length === 0
      || verificationUrl === undefined) {
      throw new Error(`${provider} device-code response is missing device_code/user_code/verification_uri`)
    }
    const intervalSec = typeof wire.interval === 'number' && wire.interval > 0
      ? wire.interval
      : DEFAULT_INTERVAL_SEC
    const expiresInSec = typeof wire.expires_in === 'number' && wire.expires_in > 0
      ? wire.expires_in
      : DEFAULT_EXPIRES_IN_SEC

    const controller = new AbortController()
    let resolveToken!: (token: DeviceTokenResult) => void
    let rejectToken!: (error: Error) => void
    const tokenPromise = new Promise<DeviceTokenResult>((resolve, reject) => {
      resolveToken = resolve
      rejectToken = reject
    })
    // The promise settles exactly once, from the poll loop below; an unhandled
    // rejection must not surface if nobody awaited waitToken after a cancel.
    tokenPromise.catch(() => undefined)

    const settle = (error?: Error, token?: DeviceTokenResult): void => {
      // Identity check: a late settle from a stale attempt (its poll loop or a
      // cancel arriving after it already settled) must not kill a NEW attempt
      // the user started for the same provider.
      if (this.attempts.get(provider) !== attempt) return
      this.attempts.delete(provider)
      if (error !== undefined) rejectToken(error)
      else if (token !== undefined) resolveToken(token)
    }

    const poll = async (): Promise<void> => {
      let intervalMs = intervalSec * 1000
      const deadline = Date.now() + expiresInSec * 1000
      while (true) {
        await sleep(intervalMs, controller.signal)
        if (Date.now() >= deadline) {
          settle(new Error(`login timed out after ${String(Math.round(expiresInSec))}s`))
          return
        }
        const pollResponse = await fetchFn(spec.tokenUrl, {
          method: 'POST',
          headers,
          body: new URLSearchParams({
            client_id: spec.clientId,
            device_code: wire.device_code as string,
            grant_type: 'urn:ietf:params:oauth:grant-type:device_code',
          }).toString(),
          signal: controller.signal,
        })
        const result = await pollResponse.json() as DeviceTokenWire
        if (typeof result.access_token === 'string' && result.access_token.length > 0) {
          const expiresIn = Number(result.expires_in)
          settle(undefined, {
            accessToken: result.access_token,
            ...nonEmptyString(result.refresh_token) === undefined ? {} : { refreshToken: result.refresh_token as string },
            ...Number.isFinite(expiresIn) && expiresIn > 0 ? { expiresIn } : {},
            ...nonEmptyString(result.scope) === undefined ? {} : { scope: result.scope as string },
          })
          return
        }
        switch (result.error) {
          case 'authorization_pending':
            break
          case 'slow_down':
            // RFC 8628 §3.5: add five seconds to the poll interval.
            intervalMs += 5000
            break
          case 'access_denied':
            settle(new Error(`login declined on the ${label} authorization page`))
            return
          case 'expired_token':
            settle(new Error('the device code expired before authorization completed'))
            return
          default:
            settle(new Error(
              `${provider} device-flow polling failed: ${result.error_description ?? result.error ?? `HTTP ${String(pollResponse.status)}`}`,
            ))
            return
        }
      }
    }

    const attempt: DeviceAttempt = {
      verificationUrl,
      userCode: wire.user_code,
      waitToken: () => tokenPromise.then(token => token.accessToken),
      waitTokens: () => tokenPromise,
      cancel: () => {
        controller.abort(new Error('login cancelled'))
        settle(new Error('login cancelled'))
      },
    }
    this.attempts.set(provider, attempt)

    void poll().catch((error: unknown) => {
      // Aborts land here from sleep/fetch; everything else is a transport or
      // parse failure worth surfacing as the login failure.
      settle(error instanceof Error ? error : new Error(String(error)))
    })
    return attempt
  }
}
