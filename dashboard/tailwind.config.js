import tailwindcssAnimate from 'tailwindcss-animate'

/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ['class'],
  content: [
    './pages/**/*.{ts,tsx}',
    './components/**/*.{ts,tsx}',
    './app/**/*.{ts,tsx}',
    './src/**/*.{ts,tsx}'
  ],
  theme: {
    container: {
      center: true,
      padding: '2rem',
      screens: {
        '2xl': '1440px'
      }
    },
    extend: {
      colors: {
        // Material 3 — ICCN Design System
        'on-primary': '#ffffff',
        'primary-fixed-dim': '#ffb5a0',
        'on-secondary-container': '#00723f',
        'inverse-on-surface': '#313030',
        background: '#0e0e0f',
        'on-tertiary-fixed': '#141c25',
        'on-background': '#ffffff',
        'surface-dim': '#0e0e0f',
        'surface-container-lowest': '#0e0e0e',
        'on-surface': '#ffffff',
        'surface-bright': '#3a3939',
        outline: '#ac897f',
        'surface-variant': '#353534',
        'error-container': '#93000a',
        'on-tertiary-container': '#222a34',
        'primary-container': '#e72c0e',
        secondary: '#f5fff3',
        'on-primary-fixed-variant': '#872100',
        surface: '#131314',
        'outline-variant': '#5c4038',
        'tertiary-container': '#89919d',
        tertiary: '#bfc7d4',
        'primary-fixed': '#ffdbd1',
        'secondary-fixed': '#5bffa1',
        'inverse-primary': '#b12e00',
        'secondary-fixed-dim': '#00e383',
        'on-error-container': '#ffdad6',
        'on-secondary-fixed-variant': '#00522c',
        'on-secondary': '#00391d',
        'surface-container-high': '#2a2a2a',
        primary: '#ffb5a0',
        'inverse-surface': '#e5e2e1',
        'tertiary-fixed': '#dbe3f0',
        'on-tertiary': '#29313b',
        error: '#ffb4ab',
        'surface-tint': '#ffb5a0',
        'secondary-container': '#27ff97',
        'tertiary-fixed-dim': '#bfc7d4',
        'on-primary-container': '#ffffff',
        'surface-container-highest': '#353534',
        'on-error': '#690005',
        'on-surface-variant': '#e6beb3',
        'surface-container': '#201f1f',
        'surface-container-low': '#1c1b1b',
        'on-tertiary-fixed-variant': '#3f4752',
        'on-primary-fixed': '#3b0900',
        'on-secondary-fixed': '#00210e',
        'iccn-red': '#E72C0E'
      },
      borderRadius: {
        DEFAULT: '0.125rem',
        lg: '0.25rem',
        xl: '0.5rem',
        full: '0.75rem'
      },
      spacing: {
        gutter: '16px',
        unit: '4px',
        'stack-md': '24px',
        'stack-sm': '8px',
        'container-padding': '40px',
        'row-height': '64px'
      },
      fontFamily: {
        'headline': ['JetBrains Mono', 'monospace'],
        'body': ['JetBrains Mono', 'monospace'],
        'label': ['JetBrains Mono', 'monospace'],
        'status': ['JetBrains Mono', 'monospace']
      },
      fontSize: {
        'headline-md': ['32px', { lineHeight: '1.2', letterSpacing: '0.02em', fontWeight: '600' }],
        'body-md': ['14px', { lineHeight: '1.5', fontWeight: '400' }],
        'label-caps': ['12px', { lineHeight: '1', letterSpacing: '0.1em', fontWeight: '700' }],
        'headline-sm': ['24px', { lineHeight: '1.3', fontWeight: '600' }],
        'display-lg': ['48px', { lineHeight: '1.1', letterSpacing: '-0.02em', fontWeight: '700' }],
        'body-lg': ['16px', { lineHeight: '1.6', fontWeight: '400' }],
        'status-sm': ['11px', { lineHeight: '1', letterSpacing: '0.05em', fontWeight: '700' }]
      },
      maxWidth: {
        'dashboard': '1440px'
      },
      keyframes: {
        'accordion-down': {
          from: { height: '0' },
          to: { height: 'var(--radix-accordion-content-height, auto)' }
        },
        'accordion-up': {
          from: { height: 'var(--radix-accordion-content-height, auto)' },
          to: { height: '0' }
        }
      },
      animation: {
        'accordion-down': 'accordion-down 0.2s ease-out',
        'accordion-up': 'accordion-up 0.2s ease-out'
      }
    }
  },
  plugins: [tailwindcssAnimate]
}
