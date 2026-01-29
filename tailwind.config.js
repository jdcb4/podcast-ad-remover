/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./app/web/templates/**/*.html"],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Background layers - Rich, not flat
        surface: {
          base: '#0a0a0f',
          DEFAULT: '#12121a',
          elevated: '#1a1a25',
          hover: '#22222f',
        },
        // Primary - Electric Violet
        primary: {
          50: '#faf5ff',
          100: '#f3e8ff',
          200: '#e9d5ff',
          300: '#d8b4fe',
          400: '#a78bfa',
          500: '#8b5cf6',
          600: '#7c3aed',
          700: '#6d28d9',
          800: '#5b21b6',
          900: '#4c1d95',
          950: '#2e1065',
        },
        // Accent - Electric Cyan
        accent: {
          50: '#ecfeff',
          100: '#cffafe',
          200: '#a5f3fc',
          300: '#67e8f9',
          400: '#22d3ee',
          500: '#06b6d4',
          600: '#0891b2',
          700: '#0e7490',
          800: '#155e75',
          900: '#164e63',
        },
      },
      fontFamily: {
        display: ['Space Grotesk', 'system-ui', 'sans-serif'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      fontSize: {
        'fluid-xs': 'clamp(0.75rem, 0.7rem + 0.15vw, 0.8rem)',
        'fluid-sm': 'clamp(0.875rem, 0.82rem + 0.2vw, 0.95rem)',
        'fluid-base': 'clamp(1rem, 0.95rem + 0.25vw, 1.125rem)',
        'fluid-lg': 'clamp(1.125rem, 1.05rem + 0.35vw, 1.25rem)',
        'fluid-xl': 'clamp(1.25rem, 1.15rem + 0.5vw, 1.5rem)',
        'fluid-2xl': 'clamp(1.5rem, 1.3rem + 0.8vw, 2rem)',
        'fluid-3xl': 'clamp(2rem, 1.7rem + 1.2vw, 2.5rem)',
        'fluid-4xl': 'clamp(2.5rem, 2rem + 2vw, 3.5rem)',
      },
      boxShadow: {
        'glow-sm': '0 0 10px rgba(139, 92, 246, 0.2)',
        'glow-md': '0 0 20px rgba(139, 92, 246, 0.3)',
        'glow-lg': '0 4px 16px rgba(139, 92, 246, 0.25), 0 0 40px rgba(139, 92, 246, 0.15)',
        'soft-sm': '0 1px 2px rgba(0, 0, 0, 0.3)',
        'soft-md': '0 2px 4px rgba(0, 0, 0, 0.2), 0 8px 16px rgba(0, 0, 0, 0.15)',
        'soft-lg': '0 4px 8px rgba(0, 0, 0, 0.2), 0 12px 32px rgba(0, 0, 0, 0.2)',
        'lift': '0 12px 40px rgba(0, 0, 0, 0.3)',
      },
      borderRadius: {
        '2xl': '1rem',
        '3xl': '1.5rem',
        '4xl': '2rem',
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-out forwards',
        'slide-up': 'slideUp 0.4s ease-out forwards',
        'slide-in-left': 'slideInLeft 0.3s ease-out forwards',
        'pulse-glow': 'pulseGlow 2s ease-in-out infinite',
        'shimmer': 'shimmer 2s linear infinite',
        'float': 'float 6s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        slideInLeft: {
          '0%': { opacity: '0', transform: 'translateX(-20px)' },
          '100%': { opacity: '1', transform: 'translateX(0)' },
        },
        pulseGlow: {
          '0%, 100%': { boxShadow: '0 0 20px rgba(139, 92, 246, 0.2)' },
          '50%': { boxShadow: '0 0 30px rgba(139, 92, 246, 0.4)' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        },
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-primary': 'linear-gradient(135deg, #7c3aed, #06b6d4)',
        'gradient-glow': 'radial-gradient(ellipse at 50% 0%, rgba(139, 92, 246, 0.15), transparent 60%)',
        'mesh-gradient': 'radial-gradient(at 40% 20%, rgba(139, 92, 246, 0.1) 0%, transparent 50%), radial-gradient(at 80% 0%, rgba(6, 182, 212, 0.1) 0%, transparent 50%), radial-gradient(at 0% 50%, rgba(139, 92, 246, 0.05) 0%, transparent 50%)',
      },
    },
  },
  plugins: [],
}
