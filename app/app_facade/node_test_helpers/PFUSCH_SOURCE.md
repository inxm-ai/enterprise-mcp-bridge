# Vendored pfusch runtime

`pfusch.js` is copied verbatim from:

`https://raw.githubusercontent.com/MatthiasKainer/pfusch/main/pfusch.js`

Current upstream revision: `6a7c325138f71e281fa3fa979c158816f2d41810`

SHA-256: `052a2012b6bfd2f1ce5fffd67f312e78a609b0715ca1d8aa2177e722aceb2e92`

The generated browser application imports the corresponding GitHub Pages module. Generated
tests replace that import with this vendored copy so tests remain deterministic. When pfusch is
updated, refresh this file and the revision/checksum together, then run the facade tests.
