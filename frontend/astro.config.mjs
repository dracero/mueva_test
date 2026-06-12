import { defineConfig } from 'astro/config';
import react from '@astrojs/react';

// https://astro.build/config
export default defineConfig({
    integrations: [react()],
    vite: {
        server: {
            watch: {
                ignored: [
                    '**/node_modules/**',
                    '**/.venv/**',
                    '**/histopatologia_data/**',
                    '**/uploads/**',
                    '**/uploads_archived/**'
                ]
            }
        }
    }
});
