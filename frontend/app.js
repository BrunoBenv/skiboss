// skiboss/frontend/app.js

// URL del backend de FastAPI desplegado en Render.com
// DEBES REEMPLAZAR ESTO CON TU PROPIA URL DE RENDER.COM
const BACKEND_URL = "http://localhost:8000"; // Placeholder inicial, usa http://tu-servicio.onrender.com

document.addEventListener('DOMContentLoaded', () => {
    const getSignalBtn = document.getElementById('getSignalBtn');
    const outputSection = document.getElementById('output');
    const signalResultDiv = document.getElementById('signal-result');
    const symbolInput = document.getElementById('symbol');
    const timeframeSelect = document.getElementById('timeframe');
    const modelVersionSpan = document.getElementById('model-version');
    const tradingviewWidget = document.getElementById('tradingview-widget');

    // --- 1. Obtener la Versión del Modelo al cargar ---
    fetch(`${BACKEND_URL}/model-version`)
        .then(res => res.json())
        .then(data => {
            modelVersionSpan.textContent = data.version;
        })
        .catch(() => {
            modelVersionSpan.textContent = 'Offline';
        });

    // --- 2. Función Principal para Obtener la Señal ---
    getSignalBtn.addEventListener('click', async () => {
        const symbol = symbolInput.value.trim().toUpperCase();
        const tf = timeframeSelect.value;
        
        if (!symbol) {
            alert("Por favor, introduce un símbolo de activo.");
            return;
        }

        getSignalBtn.disabled = true;
        getSignalBtn.textContent = 'Analizando... 🧠';
        outputSection.classList.add('hidden');
        signalResultDiv.innerHTML = '<p>Cargando datos y ejecutando modelo...</p>';

        try {
            // Llamada al endpoint /signal del backend
            const response = await fetch(`${BACKEND_URL}/signal?symbol=${symbol}&tf=${tf}`);
            const data = await response.json();
            
            if (!response.ok) {
                // Manejo de errores de FastAPI (400, 500, 503)
                signalResultDiv.innerHTML = `<h3>❌ Error</h3><p>API Error: ${data.detail || 'Error desconocido'}</p>`;
                return;
            }

            // --- 3. Renderizar Resultados ---
            renderSignal(data, symbol, tf);
            outputSection.classList.remove('hidden');

        } catch (error) {
            console.error('Error al conectar con el backend:', error);
            signalResultDiv.innerHTML = `<h3>❌ Fallo de Conexión</h3><p>No se pudo conectar con el servicio de Render.com. Asegúrate de que el backend esté activo y la URL sea correcta.</p>`;
            outputSection.classList.remove('hidden');
        } finally {
            getSignalBtn.disabled = false;
            getSignalBtn.textContent = 'Obtener Señal';
        }
    });

    // --- 4. Función de Visualización ---
    function renderSignal(data, symbol, tf) {
        const signalClass = `signal-${data.signal}`;
        
        const html = `
            <h3 class="${signalClass}">SEÑAL: ${data.signal}</h3>
            <p><strong>Confianza (Probabilidad):</strong> ${Math.round(data.confidence * 100)}%</p>
            <table style="width: 100%; margin-top: 15px;">
                <tr>
                    <td style="width: 50%;"><strong>Stop Loss (SL):</strong></td>
                    <td class="${signalClass}" style="font-weight: bold;">${data.stop_loss === 0.0 ? 'N/A' : `$${data.stop_loss.toFixed(2)}`}</td>
                </tr>
                <tr>
                    <td><strong>Take Profit (TP):</strong></td>
                    <td class="${signalClass}" style="font-weight: bold;">${data.take_profit === 0.0 ? 'N/A' : `$${data.take_profit.toFixed(2)}`}</td>
                </tr>
            </table>
            <br>
            <h4>Explicación de la IA:</h4>
            <p>${data.explanation}</p>
        `;
        
        signalResultDiv.innerHTML = html;
        
        // Actualizar el gráfico de TradingView al nuevo símbolo
        updateTradingViewChart(symbol, tf);
    }
    
    // --- 5. Actualizar Iframe del Gráfico ---
    function updateTradingViewChart(symbol, tf) {
        // Mapeo simple de Timeframes de la API a intervalos de TradingView (60 = 1h, 240 = 4h)
        const tvInterval = {
            '1m': '1', '5m': '5', '15m': '15', '30m': '30',
            '1h': '60', '4h': '240', '1d': '1D'
        }[tf.toLowerCase()] || '60';

        // La URL de TradingView usa 'NASDAQ:SPY' o 'BINANCE:BTCUSDT'
        const tvSymbol = symbol.includes(':') ? symbol : `NASDAQ:${symbol}`; 
        
        tradingviewWidget.src = `https://s.tradingview.com/widgetembed/?frameElementId=tradingview-widget&symbol=${tvSymbol}&interval=${tvInterval}&hidesidetoolbar=0&hidelist=1&theme=dark&style=1&timezone=America%2FSantiago`;
    }

});