# Diagrama de Casos de Uso — Sistema de Análisis de Tráfico Urbano

## Actores
- **Analista de Tráfico (CTE)**

## Diagrama (PlantUML)

> Para renderizarlo, usa PlantUML (p. ej., planttext.com o un plugin en VS Code/IntelliJ).

```plantuml
@startuml
left to right direction

actor "Analista de Tráfico (CTE)" as Analista

rectangle "Sistema de Análisis de Tráfico Urbano" {

  usecase "Recibir y Validar\nFlujo de imagen" as UC1
  usecase "Identificar Ocupación\nde la Intersección" as UC2
  usecase "Visualizar Estado\nde Movilidad" as UC3

  UC1 --> UC2 : <<include>>
  UC2 --> UC3 : <<include>>
}

Analista --> UC1 : Captura estado
Analista --> UC3 : Consulta estado

note right of UC1
Objetivo: Asegurar que la entrada es viable
para el análisis.

Escenario Exitoso:
Recepción de imagen (JPEG)
con resolución operativa (>720p).

Escenario de Fallo (Problema):
Pérdida de señal de cámara o condiciones
ambientales extremas.
El sistema debe alertar:
"Cámara No Operativa".
end note

note right of UC2
Objetivo: Distinguir vehículos reales
en un entorno caótico.

Escenario Exitoso:
Conteo preciso de autos y descarte
de peatones fuera de la calzada.

Escenario de Fallo (Problema):
Oclusiones severas o falsos positivos.
El sistema debe filtrar duplicados y
reducir confianza si la visión es parcial.
end note

note right of UC3
Objetivo: Presentar información útil
para la toma de decisiones.

Escenario Exitoso:
Se muestra el conteo actual y una
clasificación de congestión vehicular.

Escenario de Fallo (Problema):
Datos incongruentes o error en la
visualización.
Mostrar alerta:
"Información no concluyente".
end note

@enduml