import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// base './': dist/ works from any directory `python3 -m http.server` is pointed at.
export default defineConfig({
  base: './',
  plugins: [react()],
})
