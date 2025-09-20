import path from 'path';

const addonPath = path.resolve(__dirname, '..', '..', '..', 'build', 'Release', 'mlx_array.node');
// eslint-disable-next-line @typescript-eslint/no-var-requires
const addon = require(addonPath);

export default addon;
