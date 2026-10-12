import { defineConfig, loadEnv } from 'vite'
import path from 'path'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  return {
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src')
      }
    },
    base: './',
    build: {
      outDir: path.resolve(__dirname, '../hybrid_rag/dashboard/static'),
      emptyOutDir: true,
      chunkSizeWarningLimit: 2000,
      rollupOptions: {
        output: {
          chunkFileNames: 'assets/[name]-[hash].js',
          entryFileNames: 'assets/[name]-[hash].js',
          assetFileNames: 'assets/[name]-[hash].[ext]'
        }
      }
    },
    server: {
      port: 5174,
      // The viewer calls the relative `api/*`; in dev the page lives at `/`, so
      // `/api` is proxied to the backend that ran `mount_dashboard`. The browser
      // sees one origin, so the backend needs no CORS.
      proxy: {
        '/api': {
          target: env.VITE_BACKEND_URL || 'http://localhost:8000/kg',
          changeOrigin: true
        }
      }
    }
  }
})