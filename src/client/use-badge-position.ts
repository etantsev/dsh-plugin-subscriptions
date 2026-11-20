import { useLayoutEffect, useState } from 'react'
import type { RefObject } from 'react'

/** Position the portal above its trigger, constrained to the viewport. */
export function badgePosition(
  anchor: Pick<DOMRect, 'left' | 'top' | 'bottom'>,
  panel: Pick<DOMRect, 'width' | 'height'>,
  viewport: { width: number; height: number }, gap: number, margin: number, topMargin = margin,
): { left: number; top: number } {
  const above = anchor.top - gap - panel.height
  const preferredTop = above >= topMargin ? above : anchor.bottom + gap
  return {
    left: Math.max(margin, Math.min(anchor.left, viewport.width - margin - panel.width)),
    top: Math.max(topMargin, Math.min(preferredTop, viewport.height - margin - panel.height)),
  }
}

/** Keep the portal below the Desktop title bar, including its safety padding. */
function overlayTopMargin(margin: number): number {
  const root = document.documentElement
  const clearance = Number.parseFloat(getComputedStyle(root).getPropertyValue('--dsh-frame-top-clearance'))
  if (Number.isNaN(clearance)) return margin
  return Math.max(margin, (root.hasAttribute('data-fullscreen') ? 0 : clearance) + 20)
}

/** Measure before paint and follow scroll, viewport and content size changes. */
export function useBadgePosition(
  open: boolean, anchorRef: RefObject<HTMLElement>, panelRef: RefObject<HTMLElement>, gap: number, margin: number,
): { left: number; top: number } | null {
  const [position, setPosition] = useState<{ left: number; top: number } | null>(null)
  useLayoutEffect(() => {
    if (!open) { setPosition(null); return }
    const anchor = anchorRef.current
    const panel = panelRef.current
    if (anchor === null || panel === null) return
    const place = () => {
      const next = badgePosition(anchor.getBoundingClientRect(), panel.getBoundingClientRect(),
        { width: window.innerWidth, height: window.innerHeight }, gap, margin, overlayTopMargin(margin))
      setPosition(previous => previous?.left === next.left && previous.top === next.top ? previous : next)
    }
    place()
    window.addEventListener('resize', place)
    window.addEventListener('scroll', place, true)
    const observer = typeof ResizeObserver === 'undefined' ? undefined : new ResizeObserver(place)
    observer?.observe(anchor)
    observer?.observe(panel)
    return () => {
      window.removeEventListener('resize', place)
      window.removeEventListener('scroll', place, true)
      observer?.disconnect()
    }
  }, [open, anchorRef, panelRef, gap, margin])
  return position
}
