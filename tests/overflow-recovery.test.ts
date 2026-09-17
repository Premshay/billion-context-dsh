import { test } from 'node:test'
import assert from 'node:assert/strict'
import { Context } from '@deepseek-ai/cordis'
import type { Agent } from '@deepseek-ai/dsh-agent'
import { Session } from '@deepseek-ai/dsh-session'
import AcpCompactionEngine, { AcpCompactionEngine as Named } from '../src/index.ts'
import { rebuildBlockLedger } from '../src/region.ts'
import { SUMMARY_FRAME_PREFIX } from '../src/messages.ts'
import { sessionEventsOf } from '../src/session-events.ts'
import { buildTextSession, appendUser, appendAssistant, longText } from './helpers.ts'

/**
 * M5 — the automatic trigger half that compaction-basic has and ACP was
 * missing: `agent/request-error` context-overflow recovery
 * (docs/dsh-porting-verification.md, architecture fact 2). ACP stays
 * model-driven for pressure — the pre-step listener only nudges — but a
 * provider-confirmed CONTEXT_WINDOW_EXCEEDED is the ONE automatic action:
 * emergency-compact the largest eligible block and answer the seam's
 * `{ kind: 'retry' }` protocol instead of letting the error rethrow.
 */

/** Minimal agent handle, same shape the tools tests use. */
function fakeAgent(session: Session, ctx: Context): Agent {
  return {
    id: session.id,
    session,
    options: { provider: 'test-provider', model: 'test-model' },
    ctx,
  } as unknown as Agent
}

/** Dispatch one request-error through the same bus the loop uses. */
function requestError(
  ctx: Context,
  agent: Agent,
  failure: { code: string; message: string },
  signal: AbortSignal = new AbortController().signal,
): Promise<{ kind: 'retry' } | undefined> {
  return ctx.waterfall(
    'agent/request-error',
    { agent, turn: 1, step: 1, provider: 'test-provider', failure, retryPolicy: undefined, signal },
    () => Promise.resolve(undefined),
  )
}

const OVERFLOW = { code: 'CONTEXT_WINDOW_EXCEEDED', message: 'input too long' }

function compactionSummaryCount(session: Session): number {
  return sessionEventsOf(session).filter((event) => event.type === 'compaction/summary').length
}

test('M5: request-error listener delegates non-overflow failures (error preserved)', async () => {
  const ctx = new Context()
  const engine = new Named(ctx, { modelContextLimit: 100000 })
  const session = buildTextSession(30)
  const action = await requestError(ctx, fakeAgent(session, ctx), { code: 'RATE_LIMITED', message: 'slow down' })
  assert.equal(action, undefined, 'a non-overflow failure must pass through to the default')
  assert.equal(compactionSummaryCount(session), 0, 'no compaction for a non-overflow failure')
  void engine
})

test('M5: context overflow emergency-compacts the largest eligible block and returns retry', async () => {
  const ctx = new Context()
  const engine = new Named(ctx, { modelContextLimit: 100000 })
  const session = buildTextSession(30)
  const before = sessionEventsOf(session)
  const surfaceBefore = session.surface.nodes.length
  const action = await requestError(ctx, fakeAgent(session, ctx), OVERFLOW)
  assert.deepEqual(action, { kind: 'retry' }, 'the listener owns recovery and asks the loop to retry')
  assert.equal(compactionSummaryCount(session), 1, 'exactly one durable compaction/summary')
  // The transaction is complete: start + summary + replacement + end.
  const types = sessionEventsOf(session).map((event) => event.type)
  assert.ok(types.includes('compaction/start'), 'compaction/start recorded')
  assert.ok(types.includes('compaction/end'), 'compaction/end recorded (no dangling start)')
  // The originals are intact in the append-only log; the surface shrank.
  assert.equal(
    sessionEventsOf(session).filter((event) => event.type === 'user/message' || event.type === 'assistant/message').length,
    before.filter((event) => event.type === 'user/message' || event.type === 'assistant/message').length + 1,
    'one checkpoint summary node added, originals retained',
  )
  assert.ok(session.surface.nodes.length < surfaceBefore, 'the surface shrank')
  const ledger = rebuildBlockLedger(sessionEventsOf(session))
  assert.equal(ledger.length, 1, 'one ledger block')
  const block = ledger[0]!
  assert.equal(block.topic, 'context-overflow recovery', 'the block is labeled as an overflow recovery')
  assert.match(block.summary, /context-overflow emergency compaction/, 'the marker summary names its origin')
  assert.ok(!block.summary.includes(SUMMARY_FRAME_PREFIX), 'engine-written marker is NOT stamped as a model-written summary')
  assert.ok(block.shadowedSeqs.length > 0, 'the block shadowed a non-empty range')
  void engine
})

test('M5: retry budget spent without progress → original error preserved (no infinite retry)', async () => {
  const ctx = new Context()
  const engine = new Named(ctx, { modelContextLimit: 100000 }) // default maxOverflowRetries: 1
  const session = buildTextSession(30)
  const agent = fakeAgent(session, ctx)
  assert.deepEqual(await requestError(ctx, agent, OVERFLOW), { kind: 'retry' })
  const summariesAfterFirst = compactionSummaryCount(session)
  const second = await requestError(ctx, agent, OVERFLOW)
  assert.equal(second, undefined, 'second consecutive overflow without progress is not retried')
  assert.equal(compactionSummaryCount(session), summariesAfterFirst, 'no second compaction while the budget is spent')
})

test('M5: assistant message progress resets the recovery budget', async () => {
  const ctx = new Context()
  const engine = new Named(ctx, { modelContextLimit: 100000 })
  const session = buildTextSession(30)
  const agent = fakeAgent(session, ctx)
  assert.deepEqual(await requestError(ctx, agent, OVERFLOW), { kind: 'retry' })
  // The host publishes assistant/message on the same bus when the model
  // responds — that is progress, so a LATER overflow gets a fresh budget.
  ctx.emit('session/event', session, { type: 'assistant/message' })
  // New conversation content grows a new compressible span (the first
  // recovery already consumed everything eligible).
  for (let index = 0; index < 10; index += 1) {
    appendUser(session, longText('more', index))
    appendAssistant(session, longText('more-reply', index), 2, index + 1)
  }
  assert.deepEqual(await requestError(ctx, agent, OVERFLOW), { kind: 'retry' })
  assert.equal(compactionSummaryCount(session), 2, 'a fresh budget allowed a second recovery')
})

test('M5: agent idle resets the recovery budget', async () => {
  const ctx = new Context()
  const engine = new Named(ctx, { modelContextLimit: 100000 })
  const session = buildTextSession(30)
  const agent = fakeAgent(session, ctx)
  assert.deepEqual(await requestError(ctx, agent, OVERFLOW), { kind: 'retry' })
  ctx.emit('agent/status', { agent, status: 'idle' })
  for (let index = 0; index < 10; index += 1) {
    appendUser(session, longText('more', index))
    appendAssistant(session, longText('more-reply', index), 2, index + 1)
  }
  assert.deepEqual(await requestError(ctx, agent, OVERFLOW), { kind: 'retry' })
  assert.equal(compactionSummaryCount(session), 2, 'idle reset the budget for a second recovery')
})

test('M5: nothing eligible to compact → error preserved without a phantom block', async () => {
  const ctx = new Context()
  const engine = new Named(ctx, { modelContextLimit: 100000 })
  // Every message sits inside the protected recent tail, so no eligible range
  // exists — the listener must hand the error back untouched.
  const session = buildTextSession(4)
  const action = await requestError(ctx, fakeAgent(session, ctx), OVERFLOW)
  assert.equal(action, undefined, 'no eligible range → the original error is preserved')
  assert.equal(compactionSummaryCount(session), 0, 'no phantom compaction')
})

test('M5: aborted turn never compacts', async () => {
  const ctx = new Context()
  const engine = new Named(ctx, { modelContextLimit: 100000 })
  const session = buildTextSession(30)
  const controller = new AbortController()
  controller.abort()
  const action = await requestError(ctx, fakeAgent(session, ctx), OVERFLOW, controller.signal)
  assert.equal(action, undefined, 'aborted turn → no retry')
  assert.equal(compactionSummaryCount(session), 0)
})

test('M5: maxOverflowRetries 0 disables the automatic recovery', async () => {
  const ctx = new Context()
  const engine = new Named(ctx, { modelContextLimit: 100000, maxOverflowRetries: 0 })
  const session = buildTextSession(30)
  const action = await requestError(ctx, fakeAgent(session, ctx), OVERFLOW)
  assert.equal(action, undefined, 'explicit zero budget → error preserved')
  assert.equal(compactionSummaryCount(session), 0)
})

test('M5: invalid maxOverflowRetries fails engine construction loudly', () => {
  assert.throws(() => new Named(new Context(), { maxOverflowRetries: -1 }), /maxOverflowRetries must be a non-negative integer/)
  assert.throws(() => new Named(new Context(), { maxOverflowRetries: 1.5 }), /maxOverflowRetries must be a non-negative integer/)
})

test('M5: AcpCompactionEngine default export carries the overflow listener too', async () => {
  const ctx = new Context()
  ctx.plugin(AcpCompactionEngine as never)
  await new Promise((resolve) => setTimeout(resolve, 20))
  const engine = ctx.compaction as Named
  assert.ok(engine instanceof Named)
  const session = buildTextSession(30)
  const action = await requestError(ctx, fakeAgent(session, ctx), OVERFLOW)
  assert.deepEqual(action, { kind: 'retry' })
  assert.equal(compactionSummaryCount(session), 1)
})
