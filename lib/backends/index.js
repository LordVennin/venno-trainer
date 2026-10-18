const S = require("../../public/js/schema");

const BACKENDS = {
  anima: require("./anima"),
  aitk: require("./aitk"),
};

function backendFor(job) {
  return BACKENDS[S.backendOf(job)];
}

module.exports = { BACKENDS, backendFor };
