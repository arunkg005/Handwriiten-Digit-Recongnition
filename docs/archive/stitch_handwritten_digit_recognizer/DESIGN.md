---
name: Cipher Graphite
colors:
  surface: '#0e1511'
  surface-dim: '#0e1511'
  surface-bright: '#343b36'
  surface-container-lowest: '#09100c'
  surface-container-low: '#161d19'
  surface-container: '#1a211d'
  surface-container-high: '#242c27'
  surface-container-highest: '#2f3632'
  on-surface: '#dde4dd'
  on-surface-variant: '#bbcabf'
  inverse-surface: '#dde4dd'
  inverse-on-surface: '#2b322d'
  outline: '#86948a'
  outline-variant: '#3c4a42'
  surface-tint: '#4edea3'
  primary: '#4edea3'
  on-primary: '#003824'
  primary-container: '#10b981'
  on-primary-container: '#00422b'
  inverse-primary: '#006c49'
  secondary: '#b9c7e0'
  on-secondary: '#233144'
  secondary-container: '#3c4a5e'
  on-secondary-container: '#abb9d2'
  tertiary: '#c4c7c9'
  on-tertiary: '#2d3133'
  tertiary-container: '#a0a3a5'
  on-tertiary-container: '#36393b'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#6ffbbe'
  primary-fixed-dim: '#4edea3'
  on-primary-fixed: '#002113'
  on-primary-fixed-variant: '#005236'
  secondary-fixed: '#d5e3fd'
  secondary-fixed-dim: '#b9c7e0'
  on-secondary-fixed: '#0d1c2f'
  on-secondary-fixed-variant: '#3a485c'
  tertiary-fixed: '#e0e3e5'
  tertiary-fixed-dim: '#c4c7c9'
  on-tertiary-fixed: '#191c1e'
  on-tertiary-fixed-variant: '#444749'
  background: '#0e1511'
  on-background: '#dde4dd'
  surface-variant: '#2f3632'
typography:
  headline-lg:
    fontFamily: Sora
    fontSize: 40px
    fontWeight: '700'
    lineHeight: '1.2'
    letterSpacing: -0.02em
  headline-lg-mobile:
    fontFamily: Sora
    fontSize: 32px
    fontWeight: '700'
    lineHeight: '1.2'
  headline-md:
    fontFamily: Sora
    fontSize: 24px
    fontWeight: '600'
    lineHeight: '1.3'
  body-lg:
    fontFamily: Geist
    fontSize: 18px
    fontWeight: '400'
    lineHeight: '1.6'
  body-md:
    fontFamily: Geist
    fontSize: 16px
    fontWeight: '400'
    lineHeight: '1.5'
  label-md:
    fontFamily: Geist
    fontSize: 14px
    fontWeight: '500'
    lineHeight: '1.4'
    letterSpacing: 0.01em
  code-sm:
    fontFamily: Geist
    fontSize: 13px
    fontWeight: '400'
    lineHeight: '1.4'
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  base: 8px
  container-padding: 24px
  gutter: 16px
  desktop-max-width: 1280px
---

## Brand & Style
The design system is centered on the intersection of human touch and machine precision. It targets developers and researchers who require an interface that feels like a high-end laboratory tool: focused, sophisticated, and responsive.

The aesthetic blends **Modern Corporate** structure with **Glassmorphism** accents. The interface relies on a deep, obsidian-like canvas to minimize eye strain during long sessions, while using high-contrast Emerald accents to draw attention to classification results and active drawing states. The emotional response is one of "calm intelligence"—where the complexity of the underlying neural network is masked by a serene, high-fidelity interface.

## Colors
The palette is rooted in a deep "Midnight Navy" (#0F172A) background to provide a canvas for the digit recognition workspace. 

- **Primary (Emerald):** Used strictly for active states, high-confidence scores, and primary actions. It avoids the "neon blue" cliché, opting for a sophisticated green that feels organic yet technical.
- **Secondary (Slate):** Utilized for surface layering and container backgrounds to create depth without relying on heavy shadows.
- **Neutral (Ghost White):** Reserved for primary text and iconography to ensure AAA accessibility against the dark backgrounds.
- **Status Colors:** Use a muted Coral for errors/low-confidence and a soft Amber for warnings.

## Typography
The typography strategy employs a dual-font approach. **Sora** provides a futuristic, geometric feel for headlines and prominent digit readouts. **Geist** handles all functional text, labels, and data visualizations, providing a technical, monospaced-adjacent clarity that resonates with AI-driven applications.

Large headings should use tight letter-spacing to maintain a "sleek" profile, while labels use slightly increased tracking for legibility against dark, low-light backgrounds.

## Layout & Spacing
The layout follows a **Fluid Grid** model with a strictly enforced 8px spacing rhythm. 

- **Desktop:** 12-column grid. The main "Drawing Canvas" typically spans 7-8 columns, with "Inference Results" and "Data Logs" occupying the remaining sidebar.
- **Tablet:** 8-column grid. Layout reflows to a stacked configuration where the drawing area is prioritized at the top.
- **Margins:** Use 24px margins for mobile and 48px for desktop to create a sense of premium "breathing room."
- **Stacking:** Elements are grouped into logical "Card Clusters" using 16px gutters to separate distinct functional areas of the app.

## Elevation & Depth
Depth is achieved through **Tonal Layering** and **Subtle Glassmorphism**. 

1. **Base Layer:** The deepest Navy (#0F172A).
2. **Content Layer (Cards):** Slightly lighter Slate (#1E293B) with a 1px border (#334155). 
3. **Glass Layer (Modals/Popovers):** Semi-transparent white (opacity 4%) with a 20px backdrop blur and a vibrant top-edge highlight.

Shadows are used sparingly. When applied, they are long, extra-diffused, and slightly tinted with the Primary Emerald color (e.g., `0 20px 40px rgba(16, 185, 129, 0.08)`) to simulate a subtle glow from the UI components.

## Shapes
The design system utilizes **Rounded** geometry (0.5rem base) to soften the technical nature of the app. 

- **Buttons & Inputs:** 0.5rem corner radius.
- **Large Cards:** 1rem (`rounded-lg`) to create a distinct container feel.
- **Canvas Area:** 1.5rem (`rounded-xl`) to distinguish the "human" drawing zone from the "machine" UI.

This level of roundedness ensures the interface feels approachable and modern, avoiding the harshness of sharp corners while remaining more professional than a fully pill-shaped "playful" aesthetic.

## Components

- **Buttons:** Primary buttons are solid Emerald with high-contrast dark text. Secondary buttons use the "Ghost" style—transparent with a 1px Slate border that glows Primary on hover.
- **Drawing Canvas:** The centerpiece. A high-contrast white-on-dark surface with a subtle 8px grid-dot background to guide the user's hand.
- **Result Chips:** Used to display top-3 digit predictions. They utilize a glassmorphic background with a color-coded percentage bar at the bottom (Emerald for >90% confidence).
- **Input Fields:** Darker than the card background to create an "inset" feel. Borders should transition from Slate to Emerald on focus.
- **Confidence Charts:** Minimalist line or bar charts using the Primary Emerald. Avoid heavy axes or gridlines; focus on the data trend.
- **Toggle Switches:** Small, tactile switches that use a subtle glow when in the 'on' state.