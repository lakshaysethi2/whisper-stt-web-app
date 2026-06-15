const CACHE_NAME = "whisper-stt-v1";
const STATIC_ASSETS = ["/", "/static/app.js", "/static/style.css", "/static/manifest.json"];

self.addEventListener("install", (e) => {
  e.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(STATIC_ASSETS))
  );
  self.skipWaiting();
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (e) => {
  if (e.request.url.includes("/api/")) return;
  e.respondWith(
    caches.match(e.request).then((r) => r || fetch(e.request))
  );
});
