---
name: Urban Logic
colors:
  surface: '#0f131c'
  surface-dim: '#0f131c'
  surface-bright: '#353943'
  surface-container-lowest: '#0a0e17'
  surface-container-low: '#181b25'
  surface-container: '#1c1f29'
  surface-container-high: '#262a34'
  surface-container-highest: '#31353f'
  on-surface: '#dfe2ef'
  on-surface-variant: '#c2c6d6'
  inverse-surface: '#dfe2ef'
  inverse-on-surface: '#2c303a'
  outline: '#8c909f'
  outline-variant: '#424754'
  surface-tint: '#adc6ff'
  primary: '#adc6ff'
  on-primary: '#002e6a'
  primary-container: '#4d8eff'
  on-primary-container: '#00285d'
  inverse-primary: '#005ac2'
  secondary: '#4edea3'
  on-secondary: '#003824'
  secondary-container: '#00a572'
  on-secondary-container: '#00311f'
  tertiary: '#ffb3ad'
  on-tertiary: '#68000a'
  tertiary-container: '#ff5451'
  on-tertiary-container: '#5c0008'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#d8e2ff'
  primary-fixed-dim: '#adc6ff'
  on-primary-fixed: '#001a42'
  on-primary-fixed-variant: '#004395'
  secondary-fixed: '#6ffbbe'
  secondary-fixed-dim: '#4edea3'
  on-secondary-fixed: '#002113'
  on-secondary-fixed-variant: '#005236'
  tertiary-fixed: '#ffdad7'
  tertiary-fixed-dim: '#ffb3ad'
  on-tertiary-fixed: '#410004'
  on-tertiary-fixed-variant: '#930013'
  background: '#0f131c'
  on-background: '#dfe2ef'
  surface-variant: '#31353f'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 56px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
    letterSpacing: -0.01em
  headline-lg-mobile:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  data-mono:
    fontFamily: JetBrains Mono
    fontSize: 14px
    fontWeight: '500'
    lineHeight: 20px
  label-sm:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '600'
    lineHeight: 16px
    letterSpacing: 0.05em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  unit: 4px
  gutter: 16px
  margin-mobile: 16px
  margin-desktop: 32px
  container-padding: 24px
---

## Brand & Style
The design system is engineered for high-density information environments where precision is the primary metric of trust. It targets urban planners, researchers, and data scientists who require a tool that feels like a professional instrument rather than a consumer application.

The aesthetic combines **Minimalism** with subtle **Glassmorphism**. By using deep, light-absorbing backgrounds and semi-transparent "glass" overlays for interactive panels, the UI creates a multi-layered workspace that prioritizes content over container. Every element is intentional, reflecting a scientific and authoritative tone through strict grid alignment and refined visual feedback.

## Colors
The palette is built on a "Dark Mode First" philosophy to reduce eye strain during prolonged data analysis. 

- **Surfaces**: The base layer uses Dark Navy (#0A0E17) to provide a deep anchor. Containers use Deep Slate (#1C2533) with low-opacity borders to differentiate information modules.
- **Data Accents**: Electric Blue is reserved for primary actions and active states. Emerald Green and Ruby Red function as semantic indicators for positive growth and critical thresholds respectively.
- **Typography**: High contrast is maintained with Off-white for readability, while Muted Gray is utilized for metadata and labels to maintain visual hierarchy.

## Typography
The system employs a dual-font strategy. **Inter** handles all prose and structural headers, providing a modern, neutral foundation that ensures legibility at any scale. **JetBrains Mono** is used exclusively for data points, coordinates, parameters, and status labels. This distinction helps users instantly differentiate between "system interface" and "research data." All headings should use slight negative letter-spacing to appear tighter and more authoritative.

## Layout & Spacing
This design system utilizes a **fixed 12-column grid** for desktop dashboards and a **fluid 4-column grid** for mobile. 

The spacing rhythm is based on a **4px base unit**. Component internal padding should strictly follow this scale (e.g., 8px, 12px, 16px, 24px). For data-heavy layouts, use "Compact" spacing (8px gutters) to maximize information density. For high-level reporting views, use "Standard" spacing (24px gutters) to allow the data visualization breathing room. Large dashboard modules should have a consistent 24px internal padding.

## Elevation & Depth
Depth is expressed through **Glassmorphism** and tonal layering rather than traditional shadows. 

- **Level 0 (Surface)**: Dark Navy (#0A0E17).
- **Level 1 (Container)**: Deep Slate (#1C2533) with a 1px solid border (White @ 10% opacity).
- **Level 2 (Overlays/Modals)**: Deep Slate with a 16px backdrop-blur and a subtle top-down gradient. 

Borders are the primary tool for separation. Use thin, high-precision lines (1px) to define the skeleton of the application. High-elevation elements like floating tooltips should use a 20% opacity primary color tint in their border to indicate focus.

## Shapes
The shape language is "Soft" (0.25rem radius). This maintains a professional, geometric feel that aligns with architectural and urban planning maps while avoiding the harshness of 0px corners. 

- **Standard Buttons/Inputs**: 4px radius.
- **Large Cards/Modules**: 8px radius.
- **Status Pills**: 100px (fully rounded) to contrast against the geometric grid.

## Components
- **Buttons**: Primary buttons are solid Electric Blue with Off-white text. Secondary buttons are ghost-style with a 1px Slate border and Muted Gray text that shifts to Off-white on hover.
- **Data Callouts**: Use JetBrains Mono for all numerical values. Include a 2px vertical accent bar on the left side of the value (Emerald for positive, Ruby for negative).
- **Sliders**: Track should be Muted Gray; the active fill and the circular handle should be Electric Blue. Include a floating "value callout" above the handle using the Mono font.
- **Pipeline Indicators**: A series of connected dots or chevrons. Active stages use a "glow" effect (box-shadow: 0 0 8px primary).
- **Input Fields**: Background should be a darker shade of the container color. Borders are 1px Slate, turning Electric Blue on focus. Labels must always be visible above the field in JetBrains Mono.
- **Cards**: Minimalist containers with a 1px border. Use a semi-transparent background (80% opacity) when positioned over maps or complex charts to maintain the glass effect.