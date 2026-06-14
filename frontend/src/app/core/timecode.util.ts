/** Seconds -> "H:MM:SS" (or "M:SS" under an hour). Returns null for invalid input. */
export function formatTimecode(sec: number | string | null | undefined): string | null {
  if (sec === null || sec === undefined || sec === '') return null;
  const t = Math.floor(Number(sec));
  if (!Number.isFinite(t) || t < 0) return null;
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const s = t % 60;
  const pad = (n: number) => String(n).padStart(2, '0');
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
}

/** Build a deep link to the source at the given timecode (YouTube gets ?t=<n>s). */
export function deepLink(rawUrl: string | null | undefined, sec: number | string | null | undefined): string | null {
  if (!rawUrl) return null;
  const t = sec === null || sec === undefined || sec === '' ? null : Math.floor(Number(sec));
  try {
    const u = new URL(rawUrl);
    if (t && Number.isFinite(t)) {
      const isYt = /youtube\.com|youtu\.be/.test(u.hostname);
      u.searchParams.set('t', isYt ? `${t}s` : String(t));
    }
    return u.toString();
  } catch {
    return rawUrl;
  }
}
