/**
 * Kimi Code provider: device-flow wiring (headers, full token response),
 * login/refresh session shaping, model discovery, usage parsing, and the
 * chat request body (thinking switch, reasoning replay). No network.
 */

import { test, after } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import './keep-alive.js'
import { DeviceFlowManager } from '../src/auth/device-flow.js'
import { toChatMessages } from '../src/translate/chat-completions.js'
import type { GenerateOptions } from '@deepseek-ai/dsh-llm'
import {
  completeKimiLogin,
  fetchKimiModels,
  isKimiPermanentRefreshError,
  kimiChatRequestBody,
  kimiDeviceFlow,
  KIMI_CLIENT_ID,
  KIMI_DEVICE_CODE_URL,
  KIMI_MODELS_URL,
  KIMI_TOKEN_URL,
  kimiEndpoints,
  parseKimiUsage,
  refreshKimi,
} from '../src/providers/kimi.js'
import { OAuthEndpointError } from '../src/providers/common.js'
import { accountKeyOf } from '../src/auth/store.js'

// The device id is persisted under the harness home; keep it out of the real one.
const home = mkdtempSync(join(tmpdir(), 'kimi-spec-dsh-'))
const savedHome = process.env.DSH_HOME
process.env.DSH_HOME = home
after(() => {
  if (savedHome === undefined) delete process.env.DSH_HOME
  else process.env.DSH_HOME = savedHome
  rmSync(home, { recursive: true, force: true })
})

function jwt(claims: Record<string, unknown>): string {
  return `h.${Buffer.from(JSON.stringify(claims)).toString('base64url')}.s`
}

test('kimi device flow: form-encoded client id, X-Msh headers, full token response', async () => {
  const seen: { url: string; headers: Record<string, string>; body: string }[] = []
  const fetchFn = ((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input)
    seen.push({ url, headers: init?.headers as Record<string, string>, body: String(init?.body) })
    if (url === KIMI_DEVICE_CODE_URL) {
      return Promise.resolve(new Response(JSON.stringify({
        device_code: 'dc', user_code: 'WXYZ-1234', verification_uri: '',
        verification_uri_complete: 'https://www.kimi.com/code/authorize_device?user_code=WXYZ-1234',
        interval: 0.01, expires_in: 60,
      })))
    }
    return Promise.resolve(new Response(JSON.stringify({
      access_token: 'at', refresh_token: 'rt', expires_in: 900, scope: 'kimi-code', token_type: 'Bearer',
    })))
  }) as typeof fetch
  const manager = new DeviceFlowManager()
  const attempt = await manager.start('kimi', { ...kimiDeviceFlow(), fetchFn })
  // Kimi may leave verification_uri empty; the complete URL stands in.
  assert.equal(attempt.verificationUrl, 'https://www.kimi.com/code/authorize_device?user_code=WXYZ-1234')
  const tokens = await attempt.waitTokens()
  assert.deepEqual(tokens, { accessToken: 'at', refreshToken: 'rt', expiresIn: 900, scope: 'kimi-code' })

  const device = seen.find(entry => entry.url === KIMI_DEVICE_CODE_URL)!
  const params = new URLSearchParams(device.body)
  assert.equal(params.get('client_id'), KIMI_CLIENT_ID)
  assert.equal(params.has('scope'), false)
  assert.equal(device.headers['x-msh-platform'], 'kimi_cli')
  assert.match(device.headers['x-msh-device-id'], /^[0-9a-f]{32}$/)
  const poll = seen.find(entry => entry.url === KIMI_TOKEN_URL)!
  assert.equal(new URLSearchParams(poll.body).get('grant_type'), 'urn:ietf:params:oauth:grant-type:device_code')
  assert.equal(poll.headers['x-msh-device-id'], device.headers['x-msh-device-id'])
})

test('kimi login: session carries refresh token, expiry, and claim identity', () => {
  const before = Date.now()
  const session = completeKimiLogin({ accessToken: jwt({ user_id: 'u-42' }), refreshToken: 'rt', expiresIn: 900 })
  assert.equal(session.refreshToken, 'rt')
  assert.equal(session.account, 'u-42')
  assert.ok(session.expiresAt >= before + 899_000)
  assert.throws(() => completeKimiLogin({ accessToken: 'at' }), /refresh_token/)
})

test('kimi refresh: rotates the refresh token; 401 is permanent', async () => {
  let body = ''
  const ok = ((_: unknown, init?: RequestInit) => {
    body = String(init?.body)
    return Promise.resolve(new Response(JSON.stringify({ access_token: 'at2', refresh_token: 'rt2', expires_in: 900 })))
  }) as typeof fetch
  const next = await refreshKimi({ accessToken: 'at1', refreshToken: 'rt1', expiresAt: 0, account: 'me' }, ok)
  assert.equal(new URLSearchParams(body).get('grant_type'), 'refresh_token')
  assert.equal(new URLSearchParams(body).get('refresh_token'), 'rt1')
  assert.equal(next.accessToken, 'at2')
  assert.equal(next.refreshToken, 'rt2')
  assert.equal(next.account, 'me')

  const denied = (() => Promise.resolve(new Response(JSON.stringify({ error: 'invalid_grant' }), { status: 401 }))) as typeof fetch
  const error = await refreshKimi({ accessToken: 'a', refreshToken: 'r', expiresAt: 0 }, denied).catch((e: unknown) => e)
  assert.ok(error instanceof OAuthEndpointError)
  assert.equal(isKimiPermanentRefreshError(error), true)
  assert.equal(isKimiPermanentRefreshError(new OAuthEndpointError('x', 503)), false)
})

test('kimi models: context, vision, and thinking efforts from /models', async () => {
  const fetchFn = ((input: RequestInfo | URL, init?: RequestInit) => {
    assert.equal(String(input), KIMI_MODELS_URL)
    assert.equal((init?.headers as Record<string, string>).authorization, 'Bearer at')
    return Promise.resolve(new Response(JSON.stringify({
      data: [
        { id: 'kimi-for-coding', display_name: 'Kimi for Coding', context_length: 262144, supports_reasoning: true, supports_image_in: true },
        { id: 'plain-model', context_length: 8192 },
      ],
    })))
  }) as typeof fetch
  const models = await fetchKimiModels({ accessToken: 'at', refreshToken: 'rt', expiresAt: 0 }, fetchFn)
  assert.equal(models[0].name, 'Kimi for Coding')
  assert.equal(models[0].contextWindow, 262144)
  assert.deepEqual(models[0].inputModalities, ['text', 'image'])
  assert.deepEqual(models[0].reasoning?.efforts.map(effort => effort.id), ['none', 'high'])
  assert.equal(models[1].reasoning, undefined)
  assert.deepEqual(models[1].inputModalities, ['text'])
})

test('kimi usage: rolling limits and the weekly summary become windows', () => {
  const usage = parseKimiUsage({
    usage: { limit: 100, used: 25, resetTime: '2030-01-01T00:00:00Z' },
    limits: [{ window: { duration: 300, timeUnit: 'TIME_UNIT_MINUTE' }, detail: { limit: 50, remaining: 10 } }],
  })
  assert.equal(usage.supported, true)
  assert.deepEqual(usage.windows?.map(window => [window.kind, window.usedPercent]), [['session', 80], ['weekly', 25]])
  assert.equal(usage.windows?.[1].resetsAt, Date.parse('2030-01-01T00:00:00Z'))
})

test('kimi request body: thinking switch, cache key, reasoning replay', () => {
  const options = { model: 'kimi-for-coding', sessionId: 's-1', maxTokens: 1000, messages: [] } as unknown as GenerateOptions
  const body = kimiChatRequestBody(options, [], false)
  assert.deepEqual(body.thinking, { type: 'disabled' })
  assert.equal(body.prompt_cache_key, 's-1')
  assert.equal(body.max_completion_tokens, 1000)
  assert.equal(kimiChatRequestBody(options, [], undefined).thinking, undefined)

  const messages = toChatMessages([{
    role: 'assistant',
    content: [
      { type: 'reasoning', text: 'plan' },
      { type: 'tool-call', id: 'c1', name: 'read', arguments: '{}' },
    ],
  }] as never, undefined, { replayReasoning: true })
  assert.equal(messages[0].reasoning_content, 'plan')
  assert.equal('content' in messages[0], false)
})

test('kimi.ai: the site selects auth and API hosts and survives refresh', async () => {
  assert.equal(kimiDeviceFlow('kimi.ai').deviceCodeUrl, 'https://auth.kimi.ai/api/oauth/device_authorization')
  assert.equal(kimiEndpoints('kimi.ai').chatUrl, 'https://api.kimi.ai/coding/v1/chat/completions')
  assert.equal(kimiEndpoints(undefined).tokenUrl, KIMI_TOKEN_URL)

  const session = completeKimiLogin({ accessToken: jwt({ user_id: 'u' }), refreshToken: 'rt', expiresIn: 900 }, 'kimi.ai')
  assert.equal(session.site, 'kimi.ai')
  // Same user id on both sites keys two distinct accounts.
  assert.equal(accountKeyOf('kimi', session), 'kimi.ai:u')
  assert.equal(accountKeyOf('kimi', completeKimiLogin({ accessToken: jwt({ user_id: 'u' }), refreshToken: 'rt' })), 'u')

  const urls: string[] = []
  const fetchFn = ((input: RequestInfo | URL) => {
    urls.push(String(input))
    return Promise.resolve(new Response(JSON.stringify({ access_token: 'at2', refresh_token: 'rt2', expires_in: 900 })))
  }) as typeof fetch
  const next = await refreshKimi(session, fetchFn)
  assert.deepEqual(urls, ['https://auth.kimi.ai/api/oauth/token'])
  assert.equal(next.site, 'kimi.ai')
})
