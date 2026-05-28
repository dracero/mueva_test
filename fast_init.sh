#!/bin/bash

# --- CONFIGURACIÓN DEL ENTORNO PARA EL .DESKTOP ---
# 1. Exportar rutas estándar del sistema
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

# 2. Cargar NVM si está instalado en tu usuario (el caso más común)
if [ -s "$HOME/.nvm/nvm.sh" ]; then
    export NVM_DIR="$HOME/.nvm"
    \. "$NVM_DIR/nvm.sh"
# Alternativa: Cargar FNM si usás ese administrador
elif [ -s "$HOME/.local/share/fnm/fnm" ]; then
    export PATH="$HOME/.local/share/fnm:$PATH"
    eval "`fnm env`"
fi
# --------------------------------------------------

PORT=4321
PORT_BACK=8000
URL="http://localhost:$PORT"
URL_BACK="http://localhost:$PORT_BACK/health"
URL_FRONT_HEALTH="http://localhost:$PORT/health.json"
APP_DIR="/home/usuario/projects/mueva_test"
APP_DIR_FRONT="/home/usuario/projects/mueva_test/frontend"

echo "Verificando estado del servidor Mueva RAG Histología..."

# Comprobar si el servidor BACKEND ya está respondiendo
if curl -s "$URL_BACK" > /dev/null; then
    echo "✅ El backend ya está corriendo."
    
    # Comprobar si el FRONTEND NO está respondiendo
    if ! curl -s "$URL_FRONT_HEALTH" > /dev/null; then
        echo "⏳ El frontend está apagado. Iniciando Astro en segundo plano..."
        cd "$APP_DIR_FRONT" || exit 1
        npm run dev &
        
        # Esperar un momento a que Astro levante
        while ! curl -s "$URL_FRONT_HEALTH" > /dev/null; do
            sleep 1
        done
        echo "✅ Frontend listo."
    else
        echo "✅ El frontend ya estaba corriendo."
    fi

    echo "🌐 Abriendo el navegador en $URL..."
    xdg-open "$URL"
    sleep 2
else
    echo "🚀 Actualizando dependencias e iniciando servidor..."
    
    # Ir al directorio de la app
    cd "$APP_DIR" || exit 1

    # 1. Asegurar que las dependencias estén al día antes de lanzar la app
    echo "📦 Comprobando paquetes con uv..."
    uv pip install --python "$APP_DIR"/.venv/bin/python3 -r pyproject.toml
    
    # 2. Lanzar proceso en segundo plano que espera al backend, enciende el front y abre el navegador
    (
        echo "⏳ Esperando a que la API responda para iniciar el frontend..."
        while ! curl -s "$URL_BACK" > /dev/null; do
            sleep 1
        done
        echo "✅ Backend listo."

        # Iniciar frontend si no responde
        if ! curl -s "$URL_FRONT_HEALTH" > /dev/null; then
            echo "🖥️  Iniciando frontend Astro..."
            cd "$APP_DIR_FRONT" || exit 1
            npm run dev &
            
            while ! curl -s "$URL_FRONT_HEALTH" > /dev/null; do
                sleep 1
            done
        fi

        echo "🚀 Todo listo. Abriendo navegador..."
        xdg-open "$URL"
    ) &
    
    # 3. Levantar la API usando el entorno virtual aislado
    echo "🖥️  Iniciando api.py..."
    cd "$APP_DIR" || exit 1
    "$APP_DIR"/.venv/bin/python3 api.py
    
    # Si el servidor se cae o el usuario lo cierra con Ctrl+C, mostrar un mensaje
    echo "🛑 Servidor Mueva API detenido."
    read -p "Presiona Enter para cerrar esta ventana..."
fi
