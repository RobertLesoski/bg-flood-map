import { defineConfig } from 'vite'

export default defineConfig({
  // Serve the project root (BG_3D_Map/) so /layers/, /layers/buildings.geojson etc. resolve correctly
  publicDir: '..',
  server: {
    port: 3000,
    open: true,
    fs: {
      // Allow Vite dev server to read files from the project root
      allow: ['..', '.']
    }
  },
  build: {
    outDir: '../dist',
    emptyOutDir: true,
    // Don't inline large GeoJSON assets
    assetsInlineLimit: 0,
    rollupOptions: {
      output: {
        manualChunks: {
          maplibre: ['maplibre-gl']
        }
      }
    }
  }
})
