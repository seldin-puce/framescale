/**
 * Project links. Leave a value empty to hide the corresponding button.
 * FrameScale AI is free software (GPL-3.0-or-later) and stays fully featured;
 * donations are optional and unlock nothing.
 */
export const LINKS = {
  /** Source repository, e.g. "https://github.com/<you>/framescale". */
  repository: "",
  /** GitHub Sponsors page, e.g. "https://github.com/sponsors/<you>". */
  sponsors: "",
  /** Ko-fi page, e.g. "https://ko-fi.com/<you>". */
  kofi: "",
} as const;

export const hasDonationLinks = Boolean(LINKS.sponsors || LINKS.kofi);
