import { describe, expect, it } from "vitest";
import { endedNotice } from "./ended";

describe("ended session notice", () => {
  it("says nothing while the session is open", () => {
    expect(endedNotice({ listedClosed: false, closeReason: null, idleMinutes: 30 })).toBeNull();
  });

  it("names the idle limit when the agent closed itself for idleness", () => {
    expect(endedNotice({ listedClosed: false, closeReason: "idle_timeout_expired", idleMinutes: 30 })).toBe(
      "This session closed after 30 minutes without activity. The transcript stays here; start a new session to keep working.",
    );
    expect(endedNotice({ listedClosed: true, closeReason: "idle_timeout_expired", idleMinutes: 60 })).toMatch(/^This session closed after an hour without/);
    expect(endedNotice({ listedClosed: true, closeReason: "idle_timeout_expired", idleMinutes: undefined })).toMatch(/after a while without activity/);
  });

  it("covers both causes when only the list knows the session closed", () => {
    expect(endedNotice({ listedClosed: true, closeReason: null, idleMinutes: 30 })).toBe(
      "This session has ended: it was closed, or it sat idle for 30 minutes. The transcript stays here; start a new session to keep working.",
    );
  });

  it("plain ended for a requested close, or when sessions never close on idle", () => {
    expect(endedNotice({ listedClosed: true, closeReason: "requested", idleMinutes: 30 })).toMatch(/^This session has ended\. The transcript/);
    expect(endedNotice({ listedClosed: true, closeReason: null, idleMinutes: 0 })).toMatch(/^This session has ended\. The transcript/);
  });
});
