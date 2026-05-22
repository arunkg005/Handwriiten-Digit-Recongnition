(function () {
  const root = document.documentElement;
  const themeToggle = document.getElementById("theme-toggle-input");
  const themeLabel = document.querySelector(".theme-toggle__label");
  const canvas = document.getElementById("draw-canvas");
  const canvasOverlay = document.getElementById("canvas-overlay");
  const placeholder = document.getElementById("canvas-placeholder");
  const predictionEmpty = document.getElementById("prediction-empty");
  const predictionMain = document.getElementById("prediction-main");
  const predictionDigit = document.getElementById("prediction-digit");
  const predictionConfidence = document.getElementById("prediction-confidence");
  const predictionStatus = document.getElementById("prediction-status");
  const predictionSubtitle = document.getElementById("prediction-subtitle");
  const sizeToolButton = document.getElementById("size-tool-button");
  const sizeToolPreview = document.getElementById("size-tool-preview");
  const sizePopover = document.getElementById("size-popover");
  const sizePopoverTitle = document.getElementById("size-popover-title");
  const sizePopoverValue = document.getElementById("size-popover-value");
  const brushSizeSlider = document.getElementById("brush-size-slider");
  const probabilityList = document.getElementById("probability-list");
  const historyTableBody = document.getElementById("history-table-body");
  const undoButton = document.getElementById("undo-button");
  const redoButton = document.getElementById("redo-button");
  const clearHistoryButton = document.getElementById("clear-history-button");
  const clearCanvasButton = document.getElementById("clear-canvas-button");
  const deleteCanvasButton = document.getElementById("delete-canvas-button");
  const selectToolButton = document.getElementById("select-tool-button");
  const penToolButton = document.getElementById("pen-tool-button");
  const eraserToolButton = document.getElementById("eraser-tool-button");
  const rerunPredictionButton = document.getElementById("rerun-prediction-button");
  const pageSectionLinks = Array.from(document.querySelectorAll("[data-section-link]"));
  const pageSections = Array.from(document.querySelectorAll("[data-section]"));
  const pipelineSteps = Array.from(document.querySelectorAll(".pipeline-step"));
  const pipelineStepTitle = document.getElementById("pipeline-step-title");
  const pipelineStepCopy = document.getElementById("pipeline-step-copy");

  const displayCtx = canvas.getContext("2d");
  const modelCanvas = document.createElement("canvas");
  modelCanvas.width = canvas.width;
  modelCanvas.height = canvas.height;
  const modelCtx = modelCanvas.getContext("2d");

  const MODEL_CANVAS_SURFACE = "#ffffff";
  const MODEL_PEN_STROKE = "#111111";
  const LIGHT_DISPLAY_SURFACE = "#ffffff";
  const DARK_DISPLAY_SURFACE = "#38403d";
  const LIGHT_DISPLAY_PEN = "#111111";
  const DARK_DISPLAY_PEN = "#f1f5f3";
  const POINTER_MOVE_THRESHOLD = 6;
  const MAX_PREDICTION_CACHE_ENTRIES = 50;

  const LIGHT_DISPLAY_SURFACE_RGB = hexToRgb(LIGHT_DISPLAY_SURFACE);
  const DARK_DISPLAY_SURFACE_RGB = hexToRgb(DARK_DISPLAY_SURFACE);
  const LIGHT_DISPLAY_PEN_RGB = hexToRgb(LIGHT_DISPLAY_PEN);
  const DARK_DISPLAY_PEN_RGB = hexToRgb(DARK_DISPLAY_PEN);

  const state = {
    drawing: false,
    strokeStarted: false,
    tool: "pen",
    penSize: 16,
    eraserSize: 16,
    lastPoint: null,
    pointerStartPoint: null,
    strokeBounds: null,
    hasDrawing: false,
    history: [],
    canvasHistory: [],
    historyIndex: -1,
    latestSelectionHintBBox: null,
    predictTimer: null,
    predicting: false,
    predictionRequestId: 0,
    regions: [],
    activeRegionId: null,
    predictionCache: new Map(),
    nextSnapshotId: 1,
  };

  function hexToRgb(hex) {
    const normalized = hex.replace("#", "");
    return {
      r: Number.parseInt(normalized.slice(0, 2), 16),
      g: Number.parseInt(normalized.slice(2, 4), 16),
      b: Number.parseInt(normalized.slice(4, 6), 16),
    };
  }

  function isLightTheme() {
    return root.classList.contains("light-theme");
  }

  function getDisplaySurfaceColor() {
    return isLightTheme() ? LIGHT_DISPLAY_SURFACE : DARK_DISPLAY_SURFACE;
  }

  function getDisplayPenStrokeColor() {
    return isLightTheme() ? LIGHT_DISPLAY_PEN : DARK_DISPLAY_PEN;
  }

  function getDisplaySurfaceRgb() {
    return isLightTheme() ? LIGHT_DISPLAY_SURFACE_RGB : DARK_DISPLAY_SURFACE_RGB;
  }

  function getDisplayPenRgb() {
    return isLightTheme() ? LIGHT_DISPLAY_PEN_RGB : DARK_DISPLAY_PEN_RGB;
  }

  function configureDrawingContext(context) {
    context.lineCap = "round";
    context.lineJoin = "round";
  }

  function fillCanvas(context, color) {
    context.save();
    context.globalCompositeOperation = "source-over";
    context.fillStyle = color;
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.restore();
    configureDrawingContext(context);
  }

  function renderDisplayFromModel() {
    const source = modelCtx.getImageData(0, 0, modelCanvas.width, modelCanvas.height);
    const output = displayCtx.createImageData(source.width, source.height);
    const sourceData = source.data;
    const outputData = output.data;
    const background = getDisplaySurfaceRgb();
    const stroke = getDisplayPenRgb();

    for (let index = 0; index < sourceData.length; index += 4) {
      const grayscale = sourceData[index];
      const strokeAlpha = 1 - grayscale / 255;

      outputData[index] = Math.round(background.r + (stroke.r - background.r) * strokeAlpha);
      outputData[index + 1] = Math.round(background.g + (stroke.g - background.g) * strokeAlpha);
      outputData[index + 2] = Math.round(background.b + (stroke.b - background.b) * strokeAlpha);
      outputData[index + 3] = 255;
    }

    displayCtx.putImageData(output, 0, 0);
    configureDrawingContext(displayCtx);
  }

  function renderSelectionOverlays() {
    canvasOverlay.innerHTML = "";
    state.regions.forEach((region) => {
      const overlay = document.createElement("div");
      overlay.className = `canvas-region${region.id === state.activeRegionId ? " is-active" : ""}`;
      overlay.style.left = `${region.bbox.x * 100}%`;
      overlay.style.top = `${region.bbox.y * 100}%`;
      overlay.style.width = `${region.bbox.width * 100}%`;
      overlay.style.height = `${region.bbox.height * 100}%`;
      canvasOverlay.appendChild(overlay);
    });
  }

  function setActiveSectionLink(sectionId) {
    pageSectionLinks.forEach((link) => {
      link.classList.toggle("is-active", link.dataset.sectionLink === sectionId);
    });
  }

  function syncActiveSectionLink() {
    const sectionId = window.location.hash ? window.location.hash.slice(1) : pageSections[0]?.id;
    if (sectionId) {
      setActiveSectionLink(sectionId);
    }
  }

  function initializeSectionNavigation() {
    if (pageSections.length === 0 || pageSectionLinks.length === 0) {
      return;
    }

    pageSectionLinks.forEach((link) => {
      link.addEventListener("click", () => {
        window.requestAnimationFrame(syncActiveSectionLink);
      });
    });

    const observer = new IntersectionObserver(
      (entries) => {
        const visibleEntries = entries
          .filter((entry) => entry.isIntersecting)
          .sort((first, second) => second.intersectionRatio - first.intersectionRatio);
        if (visibleEntries.length > 0) {
          setActiveSectionLink(visibleEntries[0].target.id);
        }
      },
      {
        rootMargin: "-20% 0px -55% 0px",
        threshold: [0.2, 0.35, 0.55],
      },
    );

    pageSections.forEach((section) => observer.observe(section));
    window.addEventListener("hashchange", syncActiveSectionLink);
    syncActiveSectionLink();
  }

  function setActivePipelineStep(stepButton) {
    if (!stepButton || !pipelineStepTitle || !pipelineStepCopy) {
      return;
    }

    pipelineSteps.forEach((step) => step.classList.toggle("is-active", step === stepButton));
    pipelineStepTitle.textContent = stepButton.dataset.stepTitle || "";
    pipelineStepCopy.textContent = stepButton.dataset.stepCopy || "";
  }

  function initializePipelineSteps() {
    if (pipelineSteps.length === 0) {
      return;
    }

    pipelineSteps.forEach((stepButton) => {
      stepButton.addEventListener("click", () => setActivePipelineStep(stepButton));
      stepButton.addEventListener("mouseenter", () => setActivePipelineStep(stepButton));
      stepButton.addEventListener("focus", () => setActivePipelineStep(stepButton));
    });

    setActivePipelineStep(pipelineSteps[0]);
  }

  function clearRegionState() {
    state.regions = [];
    state.activeRegionId = null;
    renderSelectionOverlays();
  }

  function applyTheme() {
    root.classList.toggle("light-theme", !themeToggle.checked);
    if (themeLabel) {
      themeLabel.textContent = themeToggle.checked ? "Dark mode" : "Light mode";
    }
    updateSizePreview();
    renderDisplayFromModel();
    renderSelectionOverlays();
  }

  function createZeroProbabilities() {
    const probabilities = {};
    for (let index = 0; index < 10; index += 1) {
      probabilities[String(index)] = 0;
    }
    return probabilities;
  }

  function renderIdlePrediction() {
    predictionEmpty.classList.remove("hidden");
    predictionMain.classList.add("hidden");
    predictionSubtitle.classList.add("hidden");
    predictionDigit.textContent = "";
    predictionConfidence.textContent = "";
    predictionStatus.textContent = "";
    predictionSubtitle.textContent = "";
  }

  function resetCanvas() {
    fillCanvas(modelCtx, MODEL_CANVAS_SURFACE);
    renderDisplayFromModel();
    placeholder.classList.remove("hidden");
    state.hasDrawing = false;
    state.latestSelectionHintBBox = null;
    state.predictionRequestId += 1;
    updatePredictionActionButton();
    clearRegionState();
    renderIdlePrediction();
    renderProbabilities(createZeroProbabilities());
  }

  function syncCanvasStateFromSnapshot(snapshot) {
    state.hasDrawing = Boolean(snapshot?.hasDrawing);
    state.latestSelectionHintBBox = snapshot?.selectionHintBBox || null;
    placeholder.classList.toggle("hidden", state.hasDrawing);
    updatePredictionActionButton();
  }

  function updateCanvasActionButtons() {
    undoButton.disabled = state.historyIndex <= 0;
    redoButton.disabled = state.historyIndex >= state.canvasHistory.length - 1;
  }

  function updatePredictionActionButton() {
    rerunPredictionButton.disabled = !state.hasDrawing || state.predicting;
    rerunPredictionButton.classList.toggle("is-loading", state.predicting);
  }

  function captureCanvasSnapshot() {
    return {
      model: modelCanvas.toDataURL("image/png"),
      hasDrawing: state.hasDrawing,
      selectionHintBBox: state.latestSelectionHintBBox,
    };
  }

  function createSnapshotKey() {
    const snapshotKey = `snapshot-${state.nextSnapshotId}`;
    state.nextSnapshotId += 1;
    return snapshotKey;
  }

  function getCurrentCanvasSnapshot() {
    const currentSnapshot = state.canvasHistory[state.historyIndex];
    if (!currentSnapshot) {
      return captureCanvasSnapshot();
    }

    return {
      ...currentSnapshot,
      selectionHintBBox: state.latestSelectionHintBBox,
    };
  }

  function commitCanvasSnapshot() {
    const snapshot = {
      ...captureCanvasSnapshot(),
      key: createSnapshotKey(),
    };
    const currentSnapshot = state.canvasHistory[state.historyIndex];
    if (
      currentSnapshot
      && currentSnapshot.model === snapshot.model
      && JSON.stringify(currentSnapshot.selectionHintBBox) === JSON.stringify(snapshot.selectionHintBBox)
    ) {
      updateCanvasActionButtons();
      return;
    }

    state.canvasHistory = state.canvasHistory.slice(0, state.historyIndex + 1);
    state.canvasHistory.push(snapshot);
    if (state.canvasHistory.length > 50) {
      state.canvasHistory.shift();
    }
    state.historyIndex = state.canvasHistory.length - 1;
    updateCanvasActionButtons();
  }

  function drawImageSnapshot(snapshot) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => {
        modelCtx.clearRect(0, 0, modelCanvas.width, modelCanvas.height);
        modelCtx.drawImage(image, 0, 0, modelCanvas.width, modelCanvas.height);
        renderDisplayFromModel();
        syncCanvasStateFromSnapshot(snapshot);
        clearRegionState();
        renderIdlePrediction();
        renderProbabilities(createZeroProbabilities());
        resolve();
      };
      image.onerror = () => reject(new Error("Could not restore canvas state."));
      image.src = snapshot.model;
    });
  }

  async function restoreCanvasHistory(index) {
    if (index < 0 || index >= state.canvasHistory.length) return;
    state.historyIndex = index;
    await drawImageSnapshot(state.canvasHistory[index]);
    updateCanvasActionButtons();
    queuePrediction();
  }

  function initializeCanvasHistory() {
    const snapshot = {
      ...captureCanvasSnapshot(),
      key: createSnapshotKey(),
    };
    state.canvasHistory = [snapshot];
    state.historyIndex = 0;
    updateCanvasActionButtons();
  }

  function resetCanvasHistory() {
    const snapshot = {
      ...captureCanvasSnapshot(),
      key: createSnapshotKey(),
    };
    state.canvasHistory = [snapshot];
    state.historyIndex = 0;
    updateCanvasActionButtons();
  }

  function updateSizeControlVisibility() {
    const isDrawingTool = state.tool === "pen" || state.tool === "eraser";
    sizeToolButton.classList.toggle("hidden", !isDrawingTool);
    if (!isDrawingTool) {
      sizePopover.classList.add("hidden");
      sizeToolButton.classList.remove("is-active");
    }
  }

  function updateCanvasCursor() {
    canvas.style.cursor = state.tool === "select" ? "pointer" : "crosshair";
  }

  function updateSizePreview() {
    const activeSize = getActiveBrushSize();
    const previewSize = Math.max(8, Math.min(18, Math.round(activeSize * 0.42)));
    const lightTheme = isLightTheme();
    sizeToolPreview.style.width = `${previewSize}px`;
    sizeToolPreview.style.height = `${previewSize}px`;
    sizeToolPreview.style.borderRadius = state.tool === "eraser" ? "0.3rem" : "999px";
    sizeToolPreview.style.background = state.tool === "eraser"
      ? (lightTheme ? "#cfd9ff" : "#dfe8ff")
      : (lightTheme ? "#314239" : "#f2f5f3");
    sizePopoverTitle.textContent = state.tool === "eraser" ? "Eraser Size" : "Pen Size";
    sizePopoverValue.textContent = `${activeSize}px`;
    brushSizeSlider.value = String(activeSize);
  }

  function setSizePopoverOpen(isOpen) {
    sizePopover.classList.toggle("hidden", !isOpen);
    sizeToolButton.classList.toggle("is-active", isOpen);
  }

  function renderProbabilities(probabilities) {
    probabilityList.innerHTML = "";
    const values = Array.from({ length: 10 }, (_, digit) => Number(probabilities[String(digit)] || 0));
    const highestProbability = Math.max(...values);

    for (let digit = 0; digit < 10; digit += 1) {
      const probability = values[digit];
      const percentValue = Math.round(probability * 100);
      const bar = document.createElement("div");
      const isPeak = highestProbability > 0 && probability === highestProbability;
      const hasValue = probability > 0;
      const valueLabel = hasValue ? `${percentValue}%` : "";
      const fillMarkup = hasValue ? `<span class="probability-bar__fill"></span>` : "";

      bar.className = `probability-bar${isPeak ? " is-peak" : ""}${hasValue ? " has-value" : " is-zero"}`;
      bar.setAttribute("aria-label", `Digit ${digit}: ${hasValue ? `${(probability * 100).toFixed(2)} percent` : "0 percent"}`);
      bar.style.setProperty("--probability-height", `${Math.max(probability * 100, 0)}%`);
      bar.innerHTML = `
        <span class="probability-bar__value">${valueLabel}</span>
        <span class="probability-bar__track">
          ${fillMarkup}
        </span>
        <span class="probability-bar__digit">${digit}</span>
      `;
      probabilityList.appendChild(bar);
    }
  }

  function getActiveRegion() {
    return state.regions.find((region) => region.id === state.activeRegionId) || null;
  }

  function clonePredictionData(data) {
    return JSON.parse(JSON.stringify(data));
  }

  function getRegionCenter(region) {
    return {
      x: region.bbox.x + region.bbox.width / 2,
      y: region.bbox.y + region.bbox.height / 2,
    };
  }

  function getIntersectionArea(firstBox, secondBox) {
    const left = Math.max(firstBox.x, secondBox.x);
    const top = Math.max(firstBox.y, secondBox.y);
    const right = Math.min(firstBox.x + firstBox.width, secondBox.x + secondBox.width);
    const bottom = Math.min(firstBox.y + firstBox.height, secondBox.y + secondBox.height);
    return Math.max(0, right - left) * Math.max(0, bottom - top);
  }

  function getActiveRegionIdFromHint(regions, selectionHintBBox, fallbackRegionId = null) {
    if (!Array.isArray(regions) || regions.length === 0) {
      return null;
    }

    if (!selectionHintBBox) {
      return fallbackRegionId && regions.some((region) => region.id === fallbackRegionId)
        ? fallbackRegionId
        : regions[regions.length - 1].id;
    }

    const hintCenter = {
      x: selectionHintBBox.x + selectionHintBBox.width / 2,
      y: selectionHintBBox.y + selectionHintBBox.height / 2,
    };

    return regions.reduce((bestRegion, region) => {
      const overlap = getIntersectionArea(region.bbox, selectionHintBBox);
      const regionCenter = getRegionCenter(region);
      const distance = Math.hypot(regionCenter.x - hintCenter.x, regionCenter.y - hintCenter.y);

      if (!bestRegion) {
        return { region, overlap, distance };
      }
      if (overlap > bestRegion.overlap) {
        return { region, overlap, distance };
      }
      if (overlap < bestRegion.overlap) {
        return bestRegion;
      }
      if (distance < bestRegion.distance) {
        return { region, overlap, distance };
      }
      if (distance > bestRegion.distance) {
        return bestRegion;
      }
      if (region.bbox.y < bestRegion.region.bbox.y) {
        return { region, overlap, distance };
      }
      if (region.bbox.y > bestRegion.region.bbox.y) {
        return bestRegion;
      }
      if (region.bbox.x < bestRegion.region.bbox.x) {
        return { region, overlap, distance };
      }
      return bestRegion;
    }, null).region.id;
  }

  function buildPredictionView(data, activeRegionId = null) {
    const baseData = clonePredictionData(data);
    const regions = Array.isArray(baseData.regions) ? baseData.regions : [];
    if (regions.length === 0) {
      baseData.active_region_id = null;
      return baseData;
    }

    const resolvedActiveRegionId = activeRegionId && regions.some((region) => region.id === activeRegionId)
      ? activeRegionId
      : regions[regions.length - 1].id;
    const activeRegion = regions.find((region) => region.id === resolvedActiveRegionId) || regions[regions.length - 1];

    return {
      ...baseData,
      digit: activeRegion.digit,
      confidence: activeRegion.confidence,
      probabilities: activeRegion.probabilities,
      status: activeRegion.status,
      processed_preview: activeRegion.processed_preview ?? null,
      active_region_id: activeRegion.id,
    };
  }

  function getPredictionCacheKey(snapshot) {
    return snapshot.key || snapshot.model;
  }

  function getCachedPrediction(snapshot) {
    const cacheKey = getPredictionCacheKey(snapshot);
    const cached = state.predictionCache.get(cacheKey);
    if (!cached) {
      return null;
    }

    state.predictionCache.delete(cacheKey);
    state.predictionCache.set(cacheKey, cached);
    return buildPredictionView(
      cached,
      getActiveRegionIdFromHint(cached.regions, snapshot.selectionHintBBox, cached.active_region_id),
    );
  }

  function storeCachedPrediction(snapshot, data) {
    const cacheKey = getPredictionCacheKey(snapshot);
    const cacheValue = buildPredictionView(
      data,
      getActiveRegionIdFromHint(data.regions, snapshot.selectionHintBBox, data.active_region_id),
    );
    state.predictionCache.delete(cacheKey);
    state.predictionCache.set(cacheKey, cacheValue);

    while (state.predictionCache.size > MAX_PREDICTION_CACHE_ENTRIES) {
      const oldestKey = state.predictionCache.keys().next().value;
      if (oldestKey === undefined) {
        break;
      }
      state.predictionCache.delete(oldestKey);
    }

    return buildPredictionView(
      cacheValue,
      getActiveRegionIdFromHint(cacheValue.regions, snapshot.selectionHintBBox, cacheValue.active_region_id),
    );
  }

  function renderActiveRegionPrediction() {
    const activeRegion = getActiveRegion();
    if (!activeRegion) {
      renderIdlePrediction();
      renderProbabilities(createZeroProbabilities());
      return;
    }

    predictionEmpty.classList.add("hidden");
    predictionMain.classList.remove("hidden");
    const statusText = activeRegion.status || "Prediction ready.";
    predictionDigit.textContent = statusText.startsWith("Not a number")
      ? "Not a number"
      : activeRegion.is_ambiguous || activeRegion.digit === null
        ? "Uncertain"
        : String(activeRegion.digit);
    predictionConfidence.textContent = `${(activeRegion.confidence * 100).toFixed(1)}% confidence`;

    if (state.regions.length > 1) {
      predictionSubtitle.classList.remove("hidden");
      predictionSubtitle.textContent = `${state.regions.length} digits detected`;
    } else {
      predictionSubtitle.classList.add("hidden");
      predictionSubtitle.textContent = "";
    }

    const statusParts = [];
    if (activeRegion.status && activeRegion.status !== "Prediction ready.") {
      statusParts.push(activeRegion.status);
    }
    predictionStatus.textContent = statusParts.length > 0 ? statusParts.join(" | ") : "Prediction ready.";
    renderProbabilities(activeRegion.probabilities);
  }

  function setActiveRegion(regionId, shouldAdoptHint) {
    if (!state.regions.some((region) => region.id === regionId)) {
      return;
    }

    state.activeRegionId = regionId;
    if (shouldAdoptHint) {
      const activeRegion = getActiveRegion();
      state.latestSelectionHintBBox = activeRegion ? { ...activeRegion.bbox } : state.latestSelectionHintBBox;
    }
    renderSelectionOverlays();
    renderActiveRegionPrediction();
  }

  function applyPredictionResponse(data) {
    state.regions = Array.isArray(data.regions) ? data.regions : [];
    state.activeRegionId = data.active_region_id;
    if (!state.activeRegionId && state.regions.length > 0) {
      state.activeRegionId = state.regions[state.regions.length - 1].id;
    }
    renderSelectionOverlays();
    renderActiveRegionPrediction();
  }

  function renderHistory() {
    historyTableBody.innerHTML = "";
    if (state.history.length === 0) {
      historyTableBody.innerHTML = `<tr class="empty-row"><td colspan="3">No history yet</td></tr>`;
      return;
    }

    for (let index = state.history.length - 1; index >= 0; index -= 1) {
      const entry = state.history[index];
      const row = document.createElement("tr");
      row.innerHTML = `
        <td>${entry.time}</td>
        <td>${entry.digit}</td>
        <td>${entry.confidence}</td>
      `;
      historyTableBody.appendChild(row);
    }
  }

  function pushHistory(data, snapshot, forceNewEntry = false) {
    if (data.digit === null || data.confidence === null) return;
    const snapshotKey = getPredictionCacheKey(snapshot);
    const existingIndex = forceNewEntry
      ? -1
      : state.history.findIndex((entry) => entry.snapshotKey === snapshotKey);
    if (existingIndex >= 0) {
      state.history[existingIndex] = {
        ...state.history[existingIndex],
        digit: String(data.digit),
        confidence: `${(data.confidence * 100).toFixed(1)}%`,
      };
      renderHistory();
      return;
    }

    const now = new Date();
    state.history = [
      ...state.history,
      {
        snapshotKey,
        time: now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
        digit: String(data.digit),
        confidence: `${(data.confidence * 100).toFixed(1)}%`,
      },
    ];
    renderHistory();
  }

  function getCanvasPoint(event) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: (event.clientX - rect.left) * scaleX,
      y: (event.clientY - rect.top) * scaleY,
    };
  }

  function createStrokeBounds(point) {
    return { minX: point.x, minY: point.y, maxX: point.x, maxY: point.y };
  }

  function expandStrokeBounds(point) {
    if (!state.strokeBounds) {
      state.strokeBounds = createStrokeBounds(point);
      return;
    }

    state.strokeBounds.minX = Math.min(state.strokeBounds.minX, point.x);
    state.strokeBounds.minY = Math.min(state.strokeBounds.minY, point.y);
    state.strokeBounds.maxX = Math.max(state.strokeBounds.maxX, point.x);
    state.strokeBounds.maxY = Math.max(state.strokeBounds.maxY, point.y);
  }

  function normalizeBounds(bounds) {
    if (!bounds) {
      return null;
    }

    const width = Math.max(bounds.maxX - bounds.minX, 1);
    const height = Math.max(bounds.maxY - bounds.minY, 1);
    return {
      x: bounds.minX / canvas.width,
      y: bounds.minY / canvas.height,
      width: width / canvas.width,
      height: height / canvas.height,
    };
  }

  function distanceBetween(firstPoint, secondPoint) {
    return Math.hypot(firstPoint.x - secondPoint.x, firstPoint.y - secondPoint.y);
  }

  function strokeLine(context, color, from, to) {
    context.strokeStyle = color;
    context.lineWidth = getActiveBrushSize();
    context.beginPath();
    context.moveTo(from.x, from.y);
    context.lineTo(to.x, to.y);
    context.stroke();
  }

  function drawLine(from, to) {
    const displayStroke = state.tool === "pen" ? getDisplayPenStrokeColor() : getDisplaySurfaceColor();
    const modelStroke = state.tool === "pen" ? MODEL_PEN_STROKE : MODEL_CANVAS_SURFACE;
    strokeLine(displayCtx, displayStroke, from, to);
    strokeLine(modelCtx, modelStroke, from, to);
  }

  function findRegionAtPoint(point) {
    const matchingRegions = state.regions.filter((region) => {
      const left = region.bbox.x * canvas.width;
      const top = region.bbox.y * canvas.height;
      const right = left + region.bbox.width * canvas.width;
      const bottom = top + region.bbox.height * canvas.height;
      return point.x >= left && point.x <= right && point.y >= top && point.y <= bottom;
    });

    if (matchingRegions.length === 0) {
      return null;
    }

    return matchingRegions.reduce((smallestRegion, region) => {
      const area = region.bbox.width * region.bbox.height;
      const smallestArea = smallestRegion.bbox.width * smallestRegion.bbox.height;
      return area < smallestArea ? region : smallestRegion;
    });
  }

  function beginStroke(point) {
    state.drawing = true;
    state.strokeStarted = false;
    state.pointerStartPoint = point;
    state.lastPoint = point;
    state.strokeBounds = createStrokeBounds(point);
  }

  function markCanvasAsDrawn() {
    if (!state.hasDrawing) {
      state.hasDrawing = true;
      placeholder.classList.add("hidden");
      updatePredictionActionButton();
    }
  }

  function queuePrediction() {
    window.clearTimeout(state.predictTimer);
    state.predictTimer = window.setTimeout(sendPrediction, 220);
  }

  async function sendPrediction(forceRefresh = false) {
    if (state.predicting) {
      return;
    }

    if (!state.hasDrawing) {
      clearRegionState();
      renderIdlePrediction();
      renderProbabilities(createZeroProbabilities());
      updatePredictionActionButton();
      return;
    }

    const snapshot = getCurrentCanvasSnapshot();
    if (forceRefresh) {
      state.predictionCache.delete(getPredictionCacheKey(snapshot));
    } else {
      const cachedPrediction = getCachedPrediction(snapshot);
      if (cachedPrediction) {
        applyPredictionResponse(cachedPrediction);
        pushHistory(cachedPrediction, snapshot);
        return;
      }
    }

    state.predicting = true;
    const requestId = state.predictionRequestId + 1;
    state.predictionRequestId = requestId;
    updatePredictionActionButton();

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: snapshot.model,
          selection_hint_bbox: state.latestSelectionHintBBox,
        }),
      });

      if (!response.ok) {
        throw new Error("Prediction failed.");
      }

      const data = await response.json();
      if (requestId !== state.predictionRequestId) {
        return;
      }
      const resolvedPrediction = storeCachedPrediction(snapshot, data);
      applyPredictionResponse(resolvedPrediction);
      pushHistory(resolvedPrediction, snapshot, forceRefresh);
    } catch (error) {
      console.error(error);
    } finally {
      state.predicting = false;
      updatePredictionActionButton();
    }
  }

  function setTool(tool) {
    state.tool = tool;
    selectToolButton.classList.toggle("is-active", tool === "select");
    penToolButton.classList.toggle("is-active", tool === "pen");
    eraserToolButton.classList.toggle("is-active", tool === "eraser");
    updateSizeControlVisibility();
    updateCanvasCursor();
    updateSizePreview();
  }

  function getActiveBrushSize() {
    return state.tool === "eraser" ? state.eraserSize : state.penSize;
  }

  function setBrushSize(size) {
    if (state.tool === "eraser") {
      state.eraserSize = size;
    } else {
      state.penSize = size;
    }
    updateSizePreview();
  }

  function resetPointerState() {
    state.drawing = false;
    state.strokeStarted = false;
    state.pointerStartPoint = null;
    state.lastPoint = null;
    state.strokeBounds = null;
  }

  function completeStroke(finalPoint, allowSelection) {
    if (!state.drawing || !state.pointerStartPoint) {
      return;
    }

    const releasePoint = finalPoint || state.lastPoint || state.pointerStartPoint;

    if (state.tool === "select") {
      const clickedRegion = allowSelection ? findRegionAtPoint(releasePoint) : null;
      if (clickedRegion) {
        resetPointerState();
        setActiveRegion(clickedRegion.id, true);
        return;
      }

      resetPointerState();
      return;
    }

    expandStrokeBounds(releasePoint);

    if (!state.strokeStarted) {
      drawLine(state.pointerStartPoint, state.pointerStartPoint);
    }

    markCanvasAsDrawn();
    state.latestSelectionHintBBox = normalizeBounds(state.strokeBounds);
    commitCanvasSnapshot();
    queuePrediction();

    resetPointerState();
  }

  canvas.addEventListener("pointerdown", (event) => {
    beginStroke(getCanvasPoint(event));
  });

  canvas.addEventListener("pointermove", (event) => {
    if (!state.drawing || !state.lastPoint || !state.pointerStartPoint) return;
    if (state.tool === "select") return;
    const nextPoint = getCanvasPoint(event);
    expandStrokeBounds(nextPoint);

    if (!state.strokeStarted && distanceBetween(state.pointerStartPoint, nextPoint) >= POINTER_MOVE_THRESHOLD) {
      state.strokeStarted = true;
      drawLine(state.pointerStartPoint, nextPoint);
      markCanvasAsDrawn();
    } else if (state.strokeStarted) {
      drawLine(state.lastPoint, nextPoint);
    }

    state.lastPoint = nextPoint;
  });

  canvas.addEventListener("pointerup", (event) => {
    completeStroke(getCanvasPoint(event), true);
  });
  canvas.addEventListener("pointerleave", () => {
    completeStroke(state.lastPoint, false);
  });
  canvas.addEventListener("pointercancel", () => {
    completeStroke(state.lastPoint, false);
  });

  selectToolButton.addEventListener("click", () => setTool("select"));
  penToolButton.addEventListener("click", () => setTool("pen"));
  eraserToolButton.addEventListener("click", () => setTool("eraser"));
  sizeToolButton.addEventListener("click", (event) => {
    event.stopPropagation();
    setSizePopoverOpen(sizePopover.classList.contains("hidden"));
  });
  brushSizeSlider.addEventListener("input", (event) => {
    setBrushSize(Number(event.target.value));
  });
  undoButton.addEventListener("click", async () => {
    await restoreCanvasHistory(state.historyIndex - 1);
  });

  redoButton.addEventListener("click", async () => {
    await restoreCanvasHistory(state.historyIndex + 1);
  });

  clearCanvasButton.addEventListener("click", () => {
    resetCanvas();
    commitCanvasSnapshot();
  });

  deleteCanvasButton.addEventListener("click", () => {
    resetCanvas();
    resetCanvasHistory();
    state.predictionCache.clear();
  });

  clearHistoryButton.addEventListener("click", () => {
    state.history = [];
    renderHistory();
  });

  rerunPredictionButton.addEventListener("click", () => {
    window.clearTimeout(state.predictTimer);
    sendPrediction(true);
  });

  document.addEventListener("pointerdown", (event) => {
    if (sizePopover.classList.contains("hidden")) return;
    if (sizePopover.contains(event.target) || sizeToolButton.contains(event.target)) return;
    setSizePopoverOpen(false);
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      setSizePopoverOpen(false);
    }
  });

  themeToggle.addEventListener("change", applyTheme);

  initializeSectionNavigation();
  initializePipelineSteps();
  applyTheme();
  setTool("pen");
  resetCanvas();
  initializeCanvasHistory();
  renderHistory();
})();
