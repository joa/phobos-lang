// @axe-core/puppeteer pulls in puppeteer as a peer; the scripts drive the system Chrome through
// puppeteer-core, so no install downloads a browser of its own.
module.exports = { skipDownload: true }
