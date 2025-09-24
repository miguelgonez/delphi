# 🏥 Guía Clínica Completa para Delphi
## Manual Paso a Paso para Profesionales de la Salud

---

### 📋 **¿Qué es Delphi?**

Delphi es una herramienta de **inteligencia artificial médica** que ayuda a predecir el futuro de la salud de los pacientes. Piensa en ella como un "cristal bola" muy sofisticado que puede analizar la historia médica de un paciente y predecir qué enfermedades podría desarrollar en el futuro.

**En términos simples:** 
- Das a Delphi historias clínicas de pacientes
- Delphi aprende patrones de enfermedades 
- Luego puede predecir riesgos futuros para nuevos pacientes

---

## 🚀 **PASO 1: Acceder a Delphi**

### Cómo entrar a la aplicación:
1. **Abre tu navegador web** (Chrome, Firefox, Safari, etc.)
2. **Escribe la dirección** que te proporcionaron en la barra de direcciones
3. **Presiona Enter**
4. **¡Listo!** Verás la pantalla principal de Delphi

### Lo que verás:
- Un **menú lateral izquierdo** con botones azules
- El **título principal**: "Delphi: Modelado de Trayectorias de Salud"
- Una **descripción** de qué hace la herramienta

---

## 📊 **PASO 2: Generar Datos de Prueba (Recomendado para principiantes)**

### ¿Por qué empezar con datos sintéticos?
Los **datos sintéticos** son pacientes "inventados" por la computadora, pero que siguen patrones médicos reales. Son perfectos para aprender sin preocuparte por la privacidad de pacientes reales.

### Cómo generar datos sintéticos:

#### 2.1 **Navegar al generador**
1. **Busca en el menú lateral** el botón que dice **"🧬 Generador Sintético"**
2. **Haz clic** en ese botón
3. **Verás una nueva pantalla** con dos columnas

#### 2.2 **Configurar tu población de prueba**
En la **columna izquierda** verás controles para configurar:

**📈 Número de Pacientes:**
- **Desliza la barra** para elegir cuántos pacientes virtuales quieres
- **Para empezar:** usa 500-1000 pacientes
- **Para pruebas rápidas:** usa 100 pacientes

**🎯 Semilla Aleatoria:**
- **Déjala en 42** (es un número mágico en programación)
- Esto garantiza que siempre obtengas los mismos resultados

**👥 Demografía:**
- **Edad Mínima:** Edad del paciente más joven (ej: 18 años)
- **Edad Máxima:** Edad del paciente mayor (ej: 85 años)
- **% Masculino:** Porcentaje de hombres en tu población (0.5 = 50%)

**⚙️ Configuraciones Predefinidas:**
Elige según tu interés clínico:
- **"diabetes_study"** - Para estudiar diabetes e hipertensión
- **"cardiovascular_study"** - Para enfermedades del corazón
- **"mental_health_study"** - Para depresión y ansiedad
- **"medium"** - Población general balanceada

#### 2.3 **Generar los datos**
1. **Haz clic** en el botón azul grande **"🧬 Generar Población Sintética"**
2. **Espera pacientemente** - verás un mensaje "Generando población sintética..."
3. **¡Éxito!** Cuando termine verás:
   - ✅ Mensaje de éxito con número de eventos médicos generados
   - 📊 **Estadísticas** de tu población (pacientes, eventos, promedio)
   - 📈 **Gráficos** mostrando distribución de enfermedades y edades
   - 📋 **Tabla** con vista previa de los datos generados

#### 2.4 **Procesar los datos para entrenamiento**
1. **Busca** el botón **"Procesar para Entrenamiento"** (aparece después de generar)
2. **Haz clic** en ese botón
3. **Espera** hasta ver: ✅ "¡Datos sintéticos procesados y listos para entrenamiento!"

---

## 🤖 **PASO 3: Entrenar tu Modelo de IA**

### ¿Qué significa "entrenar"?
**Entrenar** es enseñarle a la inteligencia artificial patrones médicos usando los datos que generaste. Es como enseñarle a un estudiante de medicina con miles de casos clínicos.

### Cómo entrenar:

#### 3.1 **Ir a la página de entrenamiento**
1. **En el menú lateral**, busca **"▲ Entrenamiento"**
2. **Haz clic** en ese botón
3. **Deberías ver**: ✅ "¡Los datos están listos para entrenamiento!"

#### 3.2 **Configurar el entrenamiento**
Verás dos columnas con opciones:

**Columna Izquierda - Configuración Básica:**
- **Número de Épocas:** Cuántas veces la IA repasa todos los datos
  - **Para pruebas:** 5-10 épocas
  - **Para resultados serios:** 20-30 épocas
- **Tamaño de Lote:** Cuántos pacientes procesa a la vez
  - **Recomendado:** 16 o 32
- **Tasa de Aprendizaje:** Qué tan rápido aprende la IA
  - **Recomendado:** Deja el valor sugerido

**Columna Derecha - Configuración Avanzada:**
- **Longitud Máxima de Secuencia:** 256 (déjalo así)
- **Número de Capas:** 6 (óptimo para la mayoría de casos)
- **Cabezas de Atención:** 8 (buena configuración por defecto)

#### 3.3 **Iniciar el entrenamiento**
1. **Haz clic** en **"🚀 Comenzar Entrenamiento"**
2. **Sé paciente** - el entrenamiento puede tomar 10-30 minutos
3. **Verás una barra de progreso** y mensajes de estado
4. **Al finalizar**: ✅ "¡Modelo entrenado exitosamente!"

---

## 📈 **PASO 4: Usar tu Modelo Entrenado**

### Analizar Trayectorias de Pacientes

#### 4.1 **Ir a Análisis de Trayectorias**
1. **En el menú lateral**, busca **"▬ Análisis de Trayectorias"**
2. **Haz clic** para ver las trayectorias generadas

#### 4.2 **Interpretar los gráficos**
- **Eje X (horizontal):** Tiempo/Edad del paciente
- **Eje Y (vertical):** Diferentes enfermedades
- **Líneas de colores:** Cada color representa un paciente diferente
- **Puntos en las líneas:** Momento cuando aparece cada enfermedad

### Predicción de Riesgos

#### 4.3 **Ir a Predicción de Riesgos**
1. **En el menú lateral**, busca **"◆ Predicción de Riesgos"**
2. **Aquí puedes**:
   - Ver probabilidades de desarrollar enfermedades específicas
   - Analizar factores de riesgo
   - Comparar diferentes perfiles de pacientes

### Interpretabilidad del Modelo

#### 4.4 **Entender cómo piensa la IA**
1. **En el menú lateral**, busca **"● Interpretabilidad"**
2. **Verás**:
   - **Mapas de atención:** Qué enfermedades considera más importantes la IA
   - **Embeddings:** Cómo agrupa enfermedades similares
   - **Patrones aprendidos:** Qué relaciones encontró entre enfermedades

---

## 📊 **PASO 5: Evaluar el Rendimiento**

### 4.5 **Ver Métricas de Rendimiento**
1. **En el menú lateral**, busca **"▌ Métricas de Rendimiento"**
2. **Interpretación de métricas importantes**:

**🎯 AUC (Área Bajo la Curva):**
- **0.5 = Aleatorio** (como tirar una moneda)
- **0.7 = Aceptable** para uso clínico básico
- **0.8 = Bueno** para decisiones médicas importantes  
- **0.9+ = Excelente** rendimiento diagnóstico

**📈 Curvas de Calibración:**
- Muestran qué tan precisas son las probabilidades predichas
- **Línea diagonal perfecta** = predicciones perfectamente calibradas

**🔍 Matriz de Confusión:**
- **Verdaderos Positivos:** Correctamente predijo enfermedad
- **Verdaderos Negativos:** Correctamente predijo sin enfermedad
- **Falsos Positivos:** Predijo enfermedad incorrectamente (falsa alarma)
- **Falsos Negativos:** No detectó enfermedad existente (más peligroso)

---

## 📁 **PASO 6: Trabajar con Datos Reales**

### Cuando estés listo para datos de pacientes reales:

#### 6.1 **Preparar tus datos**
Tus datos deben estar en **formato Excel/CSV** con estas columnas:
- **patient_id:** Identificador único (ej: PAC001, PAC002)
- **disease_name:** Nombre de la enfermedad (ej: "Diabetes", "Hipertensión")
- **age:** Edad cuando apareció la enfermedad (ej: 45.5)
- **event_date:** Fecha del diagnóstico (formato: YYYY-MM-DD, ej: 2023-01-15)
- **gender:** Género (M o F)

**Ejemplo de datos correctos:**
```
patient_id,disease_name,age,event_date,gender
PAC001,Diabetes,45.2,2023-01-15,M
PAC001,Hypertension,47.1,2023-08-22,M
PAC002,Depression,32.8,2023-03-10,F
```

#### 6.2 **Subir datos reales**
1. **Ve a** **"◦ Subir Datos"** en el menú lateral
2. **Haz clic** en **"Elige un archivo CSV"**
3. **Selecciona tu archivo** desde tu computadora
4. **Desactiva** la casilla "Usar datos sintéticos"
5. **Procesa los datos** con el botón correspondiente

---

## ⚠️ **CONSIDERACIONES IMPORTANTES**

### **Privacidad y Ética:**
- ✅ **Los datos sintéticos son 100% seguros** - no hay información real de pacientes
- ⚠️ **Con datos reales:** Asegúrate de tener permisos apropiados
- 🔒 **Nunca incluyas información identificable** (nombres, números de teléfono, direcciones)

### **Interpretación Clínica:**
- 🧠 **Delphi es una herramienta de apoyo**, no reemplaza el juicio clínico
- 📊 **Las predicciones son probabilidades**, no certezas
- 🔍 **Siempre valida resultados** con tu experiencia clínica

### **Limitaciones:**
- 📈 **La calidad de predicciones depende de la calidad de los datos**
- ⏱️ **Más datos = mejores predicciones**, pero requiere más tiempo de entrenamiento
- 🎯 **Los resultados son específicos** para la población de entrenamiento

---

## 🆘 **Resolución de Problemas Comunes**

### **Error: "No se pueden cargar los datos"**
**Solución:**
1. Verifica que tu archivo CSV tenga las columnas correctas
2. Asegúrate de que las fechas estén en formato YYYY-MM-DD
3. No uses caracteres especiales en nombres de enfermedades

### **Error: "El entrenamiento falla"**
**Solución:**
1. Reduce el número de épocas (usa 5-10)
2. Reduce el tamaño de lote (usa 8 o 16)
3. Asegúrate de tener suficientes datos (mínimo 50-100 pacientes)

### **Resultados poco confiables (AUC < 0.6)**
**Solución:**
1. Aumenta la cantidad de datos de entrenamiento
2. Incrementa el número de épocas
3. Verifica la calidad de tus datos (fechas correctas, diagnósticos precisos)

### **La aplicación va lenta**
**Solución:**
1. Reduce el número de pacientes para pruebas iniciales
2. Usa configuraciones más simples (menos capas, menos épocas)
3. Cierra otras pestañas del navegador

---

## 🎓 **Consejos para Mejores Resultados**

### **Para Investigación:**
1. **Usa poblaciones grandes** (5,000+ pacientes sintéticos)
2. **Entrena por más tiempo** (30-50 épocas)
3. **Compara diferentes configuraciones** de modelo

### **Para Uso Clínico:**
1. **Empieza con datos sintéticos** para familiarizarte
2. **Valida resultados** con casos conocidos
3. **Documenta tu metodología** para reproducibilidad

### **Para Aprendizaje:**
1. **Experimenta con diferentes tipos de estudios** (diabetes, cardiovascular, etc.)
2. **Observa cómo cambian las predicciones** con diferentes configuraciones
3. **Analiza los mapas de atención** para entender qué considera importante la IA

---

## 📞 **Contacto y Soporte**

Si tienes preguntas o necesitas ayuda adicional:
1. **Revisa esta guía** primero - tiene soluciones para la mayoría de problemas
2. **Experimenta con datos sintéticos** antes de usar datos reales
3. **Documenta cualquier error** que encuentres para reportarlo

---

**🎉 ¡Felicidades! Ahora eres capaz de usar Delphi para análisis predictivo en salud.**

**Recuerda:** La práctica hace al maestro. Empieza con datos sintéticos, experimenta con diferentes configuraciones, y gradualmente avanza hacia aplicaciones más complejas con datos reales.

---

*Esta guía fue diseñada específicamente para profesionales clínicos sin experiencia técnica previa. Cada paso ha sido probado y validado para garantizar una experiencia de aprendizaje exitosa.*