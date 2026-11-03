import { useEffect, useState } from "react";
import { backend } from "@/ipc";

const memo = new Map<string, Promise<string>>();

/** Thumbnail URL for a source file (generated once, cached on disk by the host). */
export function useThumbnail(path: string | null, ready: boolean): string | null {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!path || !ready) return;
    let live = true;
    let p = memo.get(path);
    if (!p) {
      p = backend.thumbnail(path).catch(() => "");
      memo.set(path, p);
    }
    void p.then((rel) => live && setUrl(rel ? backend.assetUrl(rel) : null));
    return () => {
      live = false;
    };
  }, [path, ready]);
  return url;
}
