import path from 'path';

function loadAddon() {
  const candidatePaths = [
    path.resolve(__dirname, '..', '..', '..', 'build', 'Release', 'mlx.node'),
    path.resolve(__dirname, '..', '..', 'build', 'Release', 'mlx.node'),
    path.resolve(process.cwd(), 'build', 'Release', 'mlx.node'),
    path.resolve(process.cwd(), 'node', 'build', 'Release', 'mlx.node'),
  ];

  for (const addonPath of candidatePaths) {
    try {
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      return require(addonPath);
    } catch (error) {
      const nodeError = error as NodeJS.ErrnoException;
      if (nodeError.code !== 'MODULE_NOT_FOUND') {
        throw error;
      }
    }
  }

  throw new Error(
    `Cannot load mlx.node from known paths. Build with "cd node && npm run build:native". Tried: ${candidatePaths.join(', ')}`,
  );
}

const mlxModule = loadAddon();

// Export the core object which contains all the functions (for backward compatibility)
export default mlxModule.core;

// Also export nn_init separately
export const nn_init = mlxModule.nn_init;
