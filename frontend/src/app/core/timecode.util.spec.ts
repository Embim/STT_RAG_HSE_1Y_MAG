import { formatTimecode, deepLink } from './timecode.util';

describe('formatTimecode', () => {
  it('formats under an hour as M:SS', () => expect(formatTimecode(754)).toBe('12:34'));
  it('formats over an hour as H:MM:SS', () => expect(formatTimecode(3725)).toBe('1:02:05'));
  it('returns null for invalid', () => {
    expect(formatTimecode(null)).toBeNull();
    expect(formatTimecode('')).toBeNull();
    expect(formatTimecode(-5)).toBeNull();
  });
});

describe('deepLink', () => {
  it('appends ?t=<n>s for youtube', () =>
    expect(deepLink('https://www.youtube.com/watch?v=abc', 754)).toContain('t=754s'));
  it('returns the url unchanged when no timecode', () =>
    expect(deepLink('https://example.com/v', null)).toBe('https://example.com/v'));
  it('returns null when no url', () => expect(deepLink(null, 10)).toBeNull());
});
