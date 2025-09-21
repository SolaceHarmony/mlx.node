"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const path_1 = __importDefault(require("path"));
const addonPath = path_1.default.resolve(__dirname, '..', '..', '..', 'build', 'Release', 'mlx_array.node');
// eslint-disable-next-line @typescript-eslint/no-var-requires
const addon = require(addonPath);
exports.default = addon;
