const path = require('path');
const addon = require(path.join(__dirname, '../../build/Release/nan_lab.node'));

console.log('hello ->', addon.hello());
console.log('zeros ->', addon.zeros([2, 2]));
