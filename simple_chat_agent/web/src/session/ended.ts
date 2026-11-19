// What the session view says once the session has ended. Pure.
//
// A session ends when its owner closes it or, since the move to Agent Kit 0.12, when it sits
// idle for `close_after_idle_minutes` (the agent publishes `agent_closed` with reason
// `idle_timeout_expired`). The transcript stays readable; new sends are refused.

export interface EndedFacts {
  /** The session list (Describe) says the workflow has closed. */
  listedClosed: boolean;
  /** The `agent_closed` reason the stream carried, if it did. */
  closeReason: string | null;
  /** The idle limit from /config, in minutes (0 or absent: sessions don't close on idle). */
  idleMinutes: number | null | undefined;
}

const KEEP_GOING = "The transcript stays here; start a new session to keep working.";

function minutes(n: number): string {
  if (n % 60 === 0) return n === 60 ? "an hour" : `${n / 60} hours`;
  return n === 1 ? "a minute" : `${n} minutes`;
}

/** The notice for an ended session, or null while it is open. */
export function endedNotice({ listedClosed, closeReason, idleMinutes }: EndedFacts): string | null {
  if (!listedClosed && closeReason === null) return null;
  const idle = idleMinutes && idleMinutes > 0 ? idleMinutes : null;
  if (closeReason === "idle_timeout_expired") {
    return idle
      ? `This session closed after ${minutes(idle)} without activity. ${KEEP_GOING}`
      : `This session closed after a while without activity. ${KEEP_GOING}`;
  }
  if (closeReason === null && idle) {
    // Ended, cause unknown here (the stream hasn't carried agent_closed): say both ways it happens.
    return `This session has ended: it was closed, or it sat idle for ${minutes(idle)}. ${KEEP_GOING}`;
  }
  return `This session has ended. ${KEEP_GOING}`;
}
