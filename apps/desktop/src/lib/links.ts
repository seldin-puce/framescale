/**
 * Project links. Leave a value empty to hide the corresponding button.
 * FrameScale AI is free software (GPL-3.0-or-later) and stays fully featured;
 * donations are optional and unlock nothing.
 */
export const LINKS = {
  /** Source repository, e.g. "https://github.com/<you>/framescale". */
  repository: "https://github.com/seldin-puce/framescale",
  /** GitHub Sponsors page, e.g. "https://github.com/sponsors/<you>". */
  sponsors: "",
  /** Ko-fi page, e.g. "https://ko-fi.com/<you>". */
  kofi: "https://ko-fi.com/A4P429YUDG",
  /** CodeFormer upstream: bundled for Face Fix under the S-Lab License 1.0 (non-commercial). */
  codeformer: "https://github.com/sczhou/CodeFormer",
} as const;

export const hasDonationLinks = Boolean(LINKS.sponsors || LINKS.kofi);
