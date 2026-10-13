/**
 * TypescriptLLM Guide - Main Application & Navigation Controller
 */

import {
  initTensorGrid,
  initRopeWidget,
  initArchitectureDiagram,
  initSamplingSandbox,
  initKVCacheWidget,
  initBf16Widget,
  initSimulatorWidget,
  initAutogradWidget,
  initTrainingLoopWidget,
  initMiniGPTWidget,
} from './widgets.js';

export const CHAPTER_LIST = [
  'fund-1', 'fund-2', 'fund-3', 'fund-4',
  'minigpt-1', 'minigpt-2', 'minigpt-3', 'minigpt-4',
  'smollm-1', 'smollm-2', 'smollm-3', 'smollm-4', 'smollm-5', 'smollm-6'
];

let currentChapterId = 'fund-1';

export function switchChapter(target) {
  let targetId = null;

  if (typeof target === 'number') {
    if (target >= 1 && target <= CHAPTER_LIST.length) {
      targetId = CHAPTER_LIST[target - 1];
    }
  } else if (typeof target === 'string') {
    if (CHAPTER_LIST.includes(target)) {
      targetId = target;
    } else {
      const parsed = parseInt(target, 10);
      if (!isNaN(parsed) && parsed >= 1 && parsed <= CHAPTER_LIST.length) {
        targetId = CHAPTER_LIST[parsed - 1];
      }
    }
  }

  if (!targetId) return;

  currentChapterId = targetId;

  // 1. Update navigation items
  document.querySelectorAll('.nav-item').forEach(item => {
    const ch = item.getAttribute('data-chapter');
    if (ch === targetId) {
      item.classList.add('active');
    } else {
      item.classList.remove('active');
    }
  });

  // 2. Update visible chapter container
  document.querySelectorAll('.chapter-container').forEach(container => {
    const ch = container.getAttribute('data-chapter');
    if (ch === targetId) {
      container.classList.add('active');
    } else {
      container.classList.remove('active');
    }
  });

  // 3. Scroll to top of main content
  window.scrollTo({ top: 0, behavior: 'smooth' });

  // 4. Initialize chapter-specific widgets
  switch (targetId) {
    case 'fund-2':
      initTensorGrid();
      break;
    case 'fund-4':
      initSamplingSandbox();
      break;
    case 'minigpt-1':
      initAutogradWidget();
      break;
    case 'minigpt-2':
      initTrainingLoopWidget();
      break;
    case 'minigpt-3':
      initMiniGPTWidget();
      break;
    case 'smollm-2':
      initKVCacheWidget();
      break;
    case 'smollm-3':
      initArchitectureDiagram();
      initRopeWidget();
      break;
    case 'smollm-4':
      initBf16Widget();
      break;
    case 'smollm-5':
      initSimulatorWidget();
      break;
  }
}

document.addEventListener('DOMContentLoaded', () => {
  // Bind sidebar nav clicks
  document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', (e) => {
      e.preventDefault();
      const ch = item.getAttribute('data-chapter');
      if (ch) switchChapter(ch);
    });
  });

  // Bind footer next/prev buttons
  document.querySelectorAll('.btn-next-chapter').forEach(btn => {
    btn.addEventListener('click', () => {
      const idx = CHAPTER_LIST.indexOf(currentChapterId);
      if (idx >= 0 && idx < CHAPTER_LIST.length - 1) {
        switchChapter(CHAPTER_LIST[idx + 1]);
      }
    });
  });

  document.querySelectorAll('.btn-prev-chapter').forEach(btn => {
    btn.addEventListener('click', () => {
      const idx = CHAPTER_LIST.indexOf(currentChapterId);
      if (idx > 0) {
        switchChapter(CHAPTER_LIST[idx - 1]);
      }
    });
  });

  // Keyboard navigation
  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    const idx = CHAPTER_LIST.indexOf(currentChapterId);
    if (e.key === 'ArrowRight' && idx >= 0 && idx < CHAPTER_LIST.length - 1) {
      switchChapter(CHAPTER_LIST[idx + 1]);
    } else if (e.key === 'ArrowLeft' && idx > 0) {
      switchChapter(CHAPTER_LIST[idx - 1]);
    }
  });

  // Initialize first chapter
  switchChapter('fund-1');
});
