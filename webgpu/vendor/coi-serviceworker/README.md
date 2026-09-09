# coi-serviceworker

Unmodified `coi-serviceworker.js` and MIT license from
[gzuidhof/coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker),
commit `7b1d2a092d0d2dd2b7270b6f12f13605de26f214` (v0.1.7).

The Pages packager places the worker beside the application. `pages_bootstrap.js`
registers it directly and waits for a controlled reload before starting Wasm.
Only the upstream service-worker branch executes; its browser registration
helper is unused. The worker adds COOP and COEP headers to fetched responses
without caching application assets.
