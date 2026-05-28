import type { APIRoute } from 'astro';

export const GET: APIRoute = async () => {
    return new Response(
        JSON.stringify({
            status: 'ok',
            mode: 'static'
        }),
        {
            status: 200,
            headers: {
                'Content-Type': 'application/json',
            },
        }
    );
};
