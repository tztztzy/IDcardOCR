import { invoke } from "@tauri-apps/api/tauri";

// 当前模式
type Mode = "correction" | "idcard" | "idcard-copy" | "ocr";
let currentMode: Mode = "correction";

// DOM 元素
const dropZone = document.getElementById("dropZone") as HTMLDivElement;
const fileInput = document.getElementById("fileInput") as HTMLInputElement;
const previewSection = document.getElementById("previewSection") as HTMLDivElement;
const previewGrid = document.getElementById("previewGrid") as HTMLDivElement;
const loading = document.getElementById("loading") as HTMLDivElement;
const loadingText = document.getElementById("loadingText") as HTMLParagraphElement;
const batchCount = document.getElementById("batchCount") as HTMLSpanElement;
const processBtn = document.getElementById("processBtn") as HTMLButtonElement;
const clearAllBtn = document.getElementById("clearAllBtn") as HTMLButtonElement;
const resultSection = document.getElementById("resultSection") as HTMLDivElement;
const resultGrid = document.getElementById("resultGrid") as HTMLDivElement;
const resultTitle = document.getElementById("resultTitle") as HTMLHeadingElement;
const exportBtn = document.getElementById("exportBtn") as HTMLButtonElement;
const progressFill = document.getElementById("progressFill") as HTMLDivElement;
const ocrStatus = document.getElementById("ocrStatus") as HTMLSpanElement;
const uploadHint = document.getElementById("uploadHint") as HTMLParagraphElement;
const menuBtns = document.querySelectorAll(".menu-btn");

// 状态
let selectedFiles: File[] = [];
let recognitionResults: RecognitionResult[] = [];
let correctionResults: CorrectionResult[] = [];
let ocrResults: OcrTextResult[] = [];

// 身份证复印件专用状态
let frontImage: File | null = null;
let backImage: File | null = null;
let pdfResult: PdfGenerateResult | null = null;

// ZIP 下载选项
let zipDownloadOptions: ZipDownloadOptions = {
  auto_rename: false,
  include_name: true,
  include_id_number: false,
};

// 初始化
async function init() {
  await checkServiceStatus();
  setupEventListeners();
}

// 检查服务状态
async function checkServiceStatus() {
  try {
    const status = await invoke<ServiceStatus>("check_ocr_status");
    if (status.ready) {
      ocrStatus.textContent = "服务就绪";
      ocrStatus.className = "status-item status-ready";
    } else {
      ocrStatus.textContent = status.message || "服务未就绪";
      ocrStatus.className = "status-item status-error";
    }
  } catch (error) {
    console.error("Failed to check service status:", error);
    ocrStatus.textContent = "服务检查失败";
    ocrStatus.className = "status-item status-error";
  }
}

interface ServiceStatus {
  ready: boolean;
  message: string;
  ocr_engine_ready: boolean;
}

interface RecognitionResult {
  filename: string;
  original_image: string;
  corrected_image: string;
  texts: string[];
  confidence: number;
  fields: IdCardFields;
}

interface IdCardFields {
  name?: string;
  gender?: string;
  ethnicity?: string;
  birth_date?: string;
  address?: string;
  id_number?: string;
}

interface CorrectionResult {
  filename: string;
  original_image: string;
  corrected_image: string;
}

interface OcrTextResult {
  filename: string;
  texts: string[];
  confidence: number;
}

interface BatchRecognitionResult {
  results: RecognitionResult[];
  total: number;
  success: number;
  failed: number;
}

interface ImageInput {
  filename: string;
  image_base64: string;
}

interface PdfGenerateResult {
  success: boolean;
  message: string;
  pdf_base64?: string;
}

interface ExcelExportResult {
  success: boolean;
  message: string;
  excel_base64?: string;
  total: number;
  written: number;
  skipped: number;
}

interface ZipDownloadOptions {
  auto_rename: boolean;
  include_name: boolean;
  include_id_number: boolean;
}

interface ZipDownloadResult {
  success: boolean;
  message: string;
  zip_base64?: string;
  filename: string;
}

interface ImageItem {
  filename: string;
  image_base64: string;
  fields?: IdCardFields;
}

function setupEventListeners() {
  // 功能菜单切换
  menuBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      menuBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      currentMode = btn.getAttribute("data-mode") as Mode;
      resetState();
      updateUIForMode();
    });
  });

  // 点击上传
  dropZone.addEventListener("click", () => {
    fileInput.click();
  });

  // 文件选择
  fileInput.addEventListener("change", (e) => {
    const files = (e.target as HTMLInputElement).files;
    if (files) {
      handleFiles(Array.from(files));
    }
  });

  // 拖拽上传
  dropZone.addEventListener("dragover", (e: DragEvent) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });

  dropZone.addEventListener("drop", (e: DragEvent) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");

    const files = e.dataTransfer?.files;
    if (files) {
      handleFiles(Array.from(files));
    }
  });

  // 处理按钮
  processBtn.addEventListener("click", processFiles);

  // 清空按钮
  clearAllBtn.addEventListener("click", () => {
    resetState();
  });

  // 导出按钮
  exportBtn.addEventListener("click", exportToExcel);
}

function updateUIForMode() {
  switch (currentMode) {
    case "correction":
      uploadHint.textContent = "支持 JPG、PNG 格式，可批量处理";
      dropZone.style.display = "block";
      break;
    case "idcard":
      uploadHint.textContent = "支持 JPG、PNG 格式，可批量处理";
      dropZone.style.display = "block";
      break;
    case "idcard-copy":
      uploadHint.textContent = "请分别上传身份证正面和反面图片";
      dropZone.style.display = "none";
      // 延迟创建上传区域，确保之前的元素已被清理
      setTimeout(() => showIdCardCopyUpload(), 0);
      break;
    case "ocr":
      uploadHint.textContent = "支持 JPG、PNG 格式，可批量处理";
      dropZone.style.display = "block";
      break;
  }
}

function resetState() {
  selectedFiles = [];
  recognitionResults = [];
  correctionResults = [];
  ocrResults = [];
  frontImage = null;
  backImage = null;
  pdfResult = null;
  previewGrid.innerHTML = "";
  previewSection.style.display = "none";
  resultSection.style.display = "none";
  updateBatchCount();
  
  // 移除身份证复印件专用上传区域
  const copyUploadSection = document.getElementById("idcardCopySection");
  if (copyUploadSection) {
    copyUploadSection.remove();
  }
  
  // 注意：dropZone 的显示由 updateUIForMode 控制
}

async function handleFiles(files: File[]) {
  const imageFiles = files.filter((f) => f.type.startsWith("image/"));

  if (imageFiles.length === 0) {
    alert("请选择图片文件");
    return;
  }

  selectedFiles = [...selectedFiles, ...imageFiles];
  updatePreview();
  updateBatchCount();
}

function updatePreview() {
  previewGrid.innerHTML = "";

  if (selectedFiles.length === 0) {
    previewSection.style.display = "none";
    return;
  }

  previewSection.style.display = "block";

  selectedFiles.forEach((file, index) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const div = document.createElement("div");
      div.className = "preview-item";
      div.innerHTML = `
        <img src="${e.target?.result}" alt="${file.name}" />
        <button class="remove-btn" data-index="${index}">×</button>
        <span class="preview-name">${file.name}</span>
      `;
      previewGrid.appendChild(div);
      
      // 绑定删除事件
      div.querySelector(".remove-btn")?.addEventListener("click", (e) => {
        e.stopPropagation();
        removeFile(index);
      });
    };
    reader.readAsDataURL(file);
  });
}

function removeFile(index: number) {
  if (index >= 0 && index < selectedFiles.length) {
    selectedFiles.splice(index, 1);
    updatePreview();
    updateBatchCount();
  }
}

function updateBatchCount() {
  batchCount.textContent = `${selectedFiles.length} 个文件`;
  processBtn.disabled = selectedFiles.length === 0;
}

async function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      resolve(result.split(",")[1]);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

async function processFiles() {
  if (selectedFiles.length === 0 && currentMode !== "idcard-copy") return;
  if (currentMode === "idcard-copy" && !frontImage && !backImage) return;

  loading.style.display = "flex";
  progressFill.style.width = "0%";

  try {
    switch (currentMode) {
      case "correction":
        await processCorrection();
        break;
      case "idcard":
        await processIdCard();
        break;
      case "idcard-copy":
        await processIdCardCopy();
        break;
      case "ocr":
        await processOcr();
        break;
    }
  } catch (error) {
    console.error("Error:", error);
    alert("处理失败: " + error);
  } finally {
    loading.style.display = "none";
  }
}

// 证件矫正
async function processCorrection() {
  correctionResults = [];

  for (let i = 0; i < selectedFiles.length; i++) {
    const file = selectedFiles[i];
    loadingText.textContent = `正在矫正 ${i + 1}/${selectedFiles.length}: ${file.name}`;
    progressFill.style.width = `${((i + 1) / selectedFiles.length) * 100}%`;

    const base64 = await fileToBase64(file);

    try {
      const result = await invoke<CorrectionResult>("correct_image", {
        imageBase64: base64,
        filename: file.name,
      });
      correctionResults.push(result);
    } catch (error) {
      console.error(`处理 ${file.name} 失败:`, error);
      // 使用原图
      correctionResults.push({
        filename: file.name,
        original_image: base64,
        corrected_image: base64,
      });
    }
  }

  displayCorrectionResults();
}

// 身份证识别 - 使用批量接口
async function processIdCard() {
  recognitionResults = [];

  loadingText.textContent = `正在准备 ${selectedFiles.length} 个文件...`;
  progressFill.style.width = "10%";

  // 将所有文件转换为 base64
  const imageInputs: ImageInput[] = [];
  for (let i = 0; i < selectedFiles.length; i++) {
    const file = selectedFiles[i];
    loadingText.textContent = `准备中 ${i + 1}/${selectedFiles.length}: ${file.name}`;
    progressFill.style.width = `${10 + (i / selectedFiles.length) * 20}%`;
    
    const base64 = await fileToBase64(file);
    imageInputs.push({
      filename: file.name,
      image_base64: base64,
    });
  }

  // 调用批量识别接口
  loadingText.textContent = `正在批量识别 ${selectedFiles.length} 个文件...`;
  progressFill.style.width = "30%";

  try {
    const batchResult = await invoke<BatchRecognitionResult>("batch_recognize_idcard", {
      images: imageInputs,
    });

    progressFill.style.width = "100%";
    loadingText.textContent = `识别完成: 成功 ${batchResult.success} 个, 失败 ${batchResult.failed} 个`;

    recognitionResults = batchResult.results;
    displayIdCardResults();
  } catch (error) {
    console.error("批量识别失败:", error);
    // 如果批量接口失败，回退到逐个处理
    loadingText.textContent = "批量接口失败，正在逐个处理...";
    progressFill.style.width = "0%";
    await processIdCardFallback();
  }
}

// 身份证识别 - 逐个处理（回退方案）
async function processIdCardFallback() {
  recognitionResults = [];

  for (let i = 0; i < selectedFiles.length; i++) {
    const file = selectedFiles[i];
    loadingText.textContent = `正在识别 ${i + 1}/${selectedFiles.length}: ${file.name}`;
    progressFill.style.width = `${((i + 1) / selectedFiles.length) * 100}%`;

    const base64 = await fileToBase64(file);

    try {
      const result = await invoke<RecognitionResult>("recognize_idcard", {
        imageBase64: base64,
        filename: file.name,
      });
      recognitionResults.push(result);
    } catch (error) {
      console.error(`处理 ${file.name} 失败:`, error);
      recognitionResults.push({
        filename: file.name,
        original_image: base64,
        corrected_image: base64,
        texts: [`处理失败: ${error}`],
        confidence: 0,
        fields: {},
      });
    }
  }

  displayIdCardResults();
}

// 通用 OCR
async function processOcr() {
  ocrResults = [];

  for (let i = 0; i < selectedFiles.length; i++) {
    const file = selectedFiles[i];
    loadingText.textContent = `正在识别 ${i + 1}/${selectedFiles.length}: ${file.name}`;
    progressFill.style.width = `${((i + 1) / selectedFiles.length) * 100}%`;

    const base64 = await fileToBase64(file);

    const result = await invoke<OcrTextResult>("ocr_image", {
      imageBase64: base64,
      filename: file.name,
    });
    ocrResults.push(result);
  }

  displayOcrResults();
}

// ========== 身份证复印件功能 ==========

/// 显示身份证复印件专用上传区域
function showIdCardCopyUpload() {
  // 检查是否已经存在
  const existingSection = document.getElementById("idcardCopySection");
  if (existingSection) {
    return;
  }

  // 获取上传区域容器
  const uploadSection = document.querySelector(".upload-section");
  if (!uploadSection) {
    return;
  }

  // 隐藏普通预览区域
  previewSection.style.display = "none";

  // 创建身份证复印件专用上传区域
  const copySection = document.createElement("div");
  copySection.id = "idcardCopySection";
  copySection.className = "idcard-copy-section";
  copySection.innerHTML = `
    <div class="idcard-upload-grid">
      <div class="idcard-upload-item">
        <p class="idcard-label">身份证正面（人像面）</p>
        <div class="idcard-upload-box" id="frontUploadBox">
          <input type="file" id="frontFileInput" accept="image/*" hidden />
          <div class="idcard-upload-placeholder" id="frontPlaceholder">
            <svg class="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="17 8 12 3 7 8"></polyline>
              <line x1="12" y1="3" x2="12" y2="15"></line>
            </svg>
            <p>点击上传正面</p>
          </div>
          <div class="idcard-preview" id="frontPreview" style="display: none;">
            <img id="frontPreviewImg" src="" alt="正面" />
            <button class="remove-btn" id="frontRemoveBtn">×</button>
          </div>
        </div>
      </div>
      <div class="idcard-upload-item">
        <p class="idcard-label">身份证反面（国徽面）</p>
        <div class="idcard-upload-box" id="backUploadBox">
          <input type="file" id="backFileInput" accept="image/*" hidden />
          <div class="idcard-upload-placeholder" id="backPlaceholder">
            <svg class="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="17 8 12 3 7 8"></polyline>
              <line x1="12" y1="3" x2="12" y2="15"></line>
            </svg>
            <p>点击上传反面</p>
          </div>
          <div class="idcard-preview" id="backPreview" style="display: none;">
            <img id="backPreviewImg" src="" alt="反面" />
            <button class="remove-btn" id="backRemoveBtn">×</button>
          </div>
        </div>
      </div>
    </div>
    <div class="idcard-action-bar">
      <button class="btn-secondary" id="clearCopyBtn">清空</button>
      <button class="btn-primary" id="generatePdfBtn" disabled>生成PDF复印件</button>
    </div>
  `;

  uploadSection.appendChild(copySection);

  // 绑定事件
  setupIdCardCopyEvents();
}

/// 设置身份证复印件上传事件
function setupIdCardCopyEvents() {
  const frontUploadBox = document.getElementById("frontUploadBox");
  const backUploadBox = document.getElementById("backUploadBox");
  const frontFileInput = document.getElementById("frontFileInput") as HTMLInputElement;
  const backFileInput = document.getElementById("backFileInput") as HTMLInputElement;
  const frontRemoveBtn = document.getElementById("frontRemoveBtn");
  const backRemoveBtn = document.getElementById("backRemoveBtn");
  const clearCopyBtn = document.getElementById("clearCopyBtn");
  const generatePdfBtn = document.getElementById("generatePdfBtn") as HTMLButtonElement;

  // 点击上传
  frontUploadBox?.addEventListener("click", (e) => {
    if (!(e.target as HTMLElement).classList.contains("remove-btn")) {
      frontFileInput?.click();
    }
  });

  backUploadBox?.addEventListener("click", (e) => {
    if (!(e.target as HTMLElement).classList.contains("remove-btn")) {
      backFileInput?.click();
    }
  });

  // 文件选择
  frontFileInput?.addEventListener("change", (e) => {
    const files = (e.target as HTMLInputElement).files;
    if (files && files.length > 0) {
      frontImage = files[0];
      updateIdCardPreview("front", files[0]);
      updateGenerateButton();
    }
  });

  backFileInput?.addEventListener("change", (e) => {
    const files = (e.target as HTMLInputElement).files;
    if (files && files.length > 0) {
      backImage = files[0];
      updateIdCardPreview("back", files[0]);
      updateGenerateButton();
    }
  });

  // 删除按钮
  frontRemoveBtn?.addEventListener("click", (e) => {
    e.stopPropagation();
    frontImage = null;
    clearIdCardPreview("front");
    updateGenerateButton();
  });

  backRemoveBtn?.addEventListener("click", (e) => {
    e.stopPropagation();
    backImage = null;
    clearIdCardPreview("back");
    updateGenerateButton();
  });

  // 清空按钮
  clearCopyBtn?.addEventListener("click", () => {
    frontImage = null;
    backImage = null;
    clearIdCardPreview("front");
    clearIdCardPreview("back");
    updateGenerateButton();
    resultSection.style.display = "none";
    pdfResult = null;
  });

  // 生成PDF按钮
  generatePdfBtn?.addEventListener("click", processFiles);
}

/// 更新身份证预览
function updateIdCardPreview(side: "front" | "back", file: File) {
  const placeholder = document.getElementById(`${side}Placeholder`);
  const preview = document.getElementById(`${side}Preview`);
  const previewImg = document.getElementById(`${side}PreviewImg`) as HTMLImageElement;

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target?.result as string;
    placeholder!.style.display = "none";
    preview!.style.display = "block";
  };
  reader.readAsDataURL(file);
}

/// 清除身份证预览
function clearIdCardPreview(side: "front" | "back") {
  const placeholder = document.getElementById(`${side}Placeholder`);
  const preview = document.getElementById(`${side}Preview`);
  const previewImg = document.getElementById(`${side}PreviewImg`) as HTMLImageElement;

  placeholder!.style.display = "flex";
  preview!.style.display = "none";
  previewImg.src = "";
}

/// 更新生成按钮状态
function updateGenerateButton() {
  const generatePdfBtn = document.getElementById("generatePdfBtn") as HTMLButtonElement;
  generatePdfBtn.disabled = !(frontImage && backImage);
}

/// 处理身份证复印件生成
async function processIdCardCopy() {
  if (!frontImage || !backImage) {
    alert("请上传身份证正反面图片");
    return;
  }

  loading.style.display = "flex";
  loadingText.textContent = "正在处理图片...";
  progressFill.style.width = "30%";

  try {
    const frontBase64 = await fileToBase64(frontImage);
    loadingText.textContent = "正在生成PDF...";
    progressFill.style.width = "60%";

    const backBase64 = await fileToBase64(backImage);
    loadingText.textContent = "正在生成PDF...";
    progressFill.style.width = "80%";

    const result = await invoke<PdfGenerateResult>("generate_idcard_copy", {
      frontImageBase64: frontBase64,
      backImageBase64: backBase64,
    });

    progressFill.style.width = "100%";

    if (result.success) {
      pdfResult = result;
      displayIdCardCopyResult();
    } else {
      alert(result.message);
    }
  } catch (error) {
    console.error("PDF生成失败:", error);
    alert("PDF生成失败: " + error);
  } finally {
    loading.style.display = "none";
  }
}

/// 显示身份证复印件结果
function displayIdCardCopyResult() {
  if (!pdfResult || !pdfResult.pdf_base64) return;

  resultTitle.textContent = "身份证复印件";
  exportBtn.style.display = "none";
  resultSection.style.display = "block";
  resultGrid.innerHTML = "";

  const div = document.createElement("div");
  div.className = "result-item pdf-result-item";
  div.innerHTML = `
    <div class="result-header">
      <span class="result-filename">身份证复印件.pdf</span>
      <span class="result-score">${pdfResult.message}</span>
    </div>
    <div class="result-body pdf-preview-container">
      <div class="pdf-preview">
        <div class="pdf-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" width="64" height="64">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
            <polyline points="14 2 14 8 20 8"></polyline>
            <line x1="16" y1="13" x2="8" y2="13"></line>
            <line x1="16" y1="17" x2="8" y2="17"></line>
            <polyline points="10 9 9 9 8 9"></polyline>
          </svg>
        </div>
        <p class="pdf-name">身份证复印件.pdf</p>
        <p class="pdf-hint">包含身份证正反面</p>
      </div>
      <button class="btn-primary download-pdf-btn" id="downloadPdfBtn">
        下载PDF文件
      </button>
    </div>
  `;

  resultGrid.appendChild(div);

  // 绑定下载事件
  const downloadBtn = document.getElementById("downloadPdfBtn");
  downloadBtn?.addEventListener("click", () => {
    if (pdfResult && pdfResult.pdf_base64) {
      downloadPdf(pdfResult.pdf_base64, "身份证复印件.pdf");
    }
  });
}

/// 下载PDF文件
function downloadPdf(base64: string, filename: string) {
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  const blob = new Blob([bytes], { type: "application/pdf" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// 显示矫正结果
function displayCorrectionResults() {
  resultTitle.textContent = "矫正结果";
  exportBtn.style.display = "none";
  resultSection.style.display = "block";
  resultGrid.innerHTML = "";

  // 添加 ZIP 下载选项和按钮
  const downloadSection = document.createElement("div");
  downloadSection.className = "zip-download-section";
  downloadSection.innerHTML = `
    <div class="zip-options">
      <label class="zip-option-label">
        <input type="checkbox" id="autoRename" ${zipDownloadOptions.auto_rename ? "checked" : ""}>
        <span>自动重命名文件</span>
      </label>
      <label class="zip-option-label" style="margin-left: 20px; opacity: ${zipDownloadOptions.auto_rename ? 1 : 0.5}">
        <input type="checkbox" id="includeName" ${zipDownloadOptions.include_name ? "checked" : ""} ${!zipDownloadOptions.auto_rename ? "disabled" : ""}>
        <span>包含姓名</span>
      </label>
      <label class="zip-option-label" style="margin-left: 20px; opacity: ${zipDownloadOptions.auto_rename ? 1 : 0.5}">
        <input type="checkbox" id="includeIdNumber" ${zipDownloadOptions.include_id_number ? "checked" : ""} ${!zipDownloadOptions.auto_rename ? "disabled" : ""}>
        <span>包含身份证号</span>
      </label>
    </div>
    <button class="btn-primary" id="downloadZipBtn">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" width="16" height="16" style="margin-right: 6px;">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
        <polyline points="7 10 12 15 17 10"></polyline>
        <line x1="12" y1="15" x2="12" y2="3"></line>
      </svg>
      下载压缩包 (${correctionResults.length}张)
    </button>
  `;
  resultSection.insertBefore(downloadSection, resultGrid);

  // 绑定选项事件
  const autoRenameCheckbox = document.getElementById("autoRename") as HTMLInputElement;
  const includeNameCheckbox = document.getElementById("includeName") as HTMLInputElement;
  const includeIdNumberCheckbox = document.getElementById("includeIdNumber") as HTMLInputElement;

  autoRenameCheckbox?.addEventListener("change", (e) => {
    zipDownloadOptions.auto_rename = (e.target as HTMLInputElement).checked;
    includeNameCheckbox.disabled = !zipDownloadOptions.auto_rename;
    includeIdNumberCheckbox.disabled = !zipDownloadOptions.auto_rename;
    (includeNameCheckbox.parentElement as HTMLElement).style.opacity = zipDownloadOptions.auto_rename ? "1" : "0.5";
    (includeIdNumberCheckbox.parentElement as HTMLElement).style.opacity = zipDownloadOptions.auto_rename ? "1" : "0.5";
  });

  includeNameCheckbox?.addEventListener("change", (e) => {
    zipDownloadOptions.include_name = (e.target as HTMLInputElement).checked;
  });

  includeIdNumberCheckbox?.addEventListener("change", (e) => {
    zipDownloadOptions.include_id_number = (e.target as HTMLInputElement).checked;
  });

  // 绑定下载按钮事件
  document.getElementById("downloadZipBtn")?.addEventListener("click", downloadAsZip);

  correctionResults.forEach((result) => {
    const div = document.createElement("div");
    div.className = "result-item correction-result-item";
    div.innerHTML = `
      <div class="result-header">
        <span class="result-filename">${escapeHtml(result.filename)}</span>
      </div>
      <div class="result-body" style="flex-direction: row; gap: 24px;">
        <div class="result-image">
          <p style="font-size: 13px; color: #666; margin-bottom: 8px;">原图</p>
          <img src="data:image/jpeg;base64,${result.original_image}" />
        </div>
        <div class="result-image">
          <p style="font-size: 13px; color: #666; margin-bottom: 8px;">矫正后</p>
          <img src="data:image/jpeg;base64,${result.corrected_image}" />
        </div>
      </div>
    `;
    resultGrid.appendChild(div);
  });
}

  // 显示身份证识别结果
function displayIdCardResults() {
  resultTitle.textContent = "身份证识别结果";
  exportBtn.style.display = recognitionResults.length > 0 ? "inline-block" : "none";
  exportBtn.textContent = "导出 Excel";
  resultSection.style.display = "block";
  resultGrid.innerHTML = "";

  recognitionResults.forEach((result) => {
    const div = document.createElement("div");
    div.className = "result-item";

    // 判断是否有提取到信息
    const hasFields = result.fields.name || result.fields.gender || result.fields.ethnicity || 
                      result.fields.birth_date || result.fields.address || result.fields.id_number;

    const fieldsHtml = hasFields ? `
      <div class="field"><span class="label">姓名:</span><span class="value">${result.fields.name || ""}</span></div>
      <div class="field"><span class="label">性别:</span><span class="value">${result.fields.gender || ""}</span></div>
      <div class="field"><span class="label">民族:</span><span class="value">${result.fields.ethnicity || ""}</span></div>
      <div class="field"><span class="label">出生:</span><span class="value">${result.fields.birth_date || ""}</span></div>
      <div class="field"><span class="label">住址:</span><span class="value">${result.fields.address || ""}</span></div>
      <div class="field"><span class="label">身份证号:</span><span class="value">${result.fields.id_number || ""}</span></div>
    ` : `
      <div class="field"><span class="label" style="color: #999;">未识别到身份证信息</span></div>
    `;

    div.innerHTML = `
      <div class="result-header">
        <span class="result-filename">${escapeHtml(result.filename)}</span>
        <span class="result-score">置信度: ${(result.confidence * 100).toFixed(1)}%</span>
      </div>
      <div class="result-body">
        <div class="result-image">
          <img src="data:image/jpeg;base64,${result.corrected_image}" alt="${escapeHtml(result.filename)}" />
        </div>
        <div class="result-content">
          <div class="result-fields">
            ${fieldsHtml}
          </div>
        </div>
      </div>
    `;

    resultGrid.appendChild(div);
  });
}

// 显示 OCR 结果
function displayOcrResults() {
  resultTitle.textContent = "OCR识别结果";
  exportBtn.style.display = "none";
  resultSection.style.display = "block";
  resultGrid.innerHTML = "";

  ocrResults.forEach((result) => {
    const div = document.createElement("div");
    div.className = "result-item ocr-result-item";

    const allText = result.texts.join("\n");

    div.innerHTML = `
      <div class="result-header">
        <span class="result-filename">${escapeHtml(result.filename)}</span>
        <span class="result-score">置信度: ${(result.confidence * 100).toFixed(1)}%</span>
      </div>
      <div class="ocr-text">${escapeHtml(allText) || "未识别到文字"}</div>
    `;

    resultGrid.appendChild(div);
  });
}

function escapeHtml(text: string): string {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

async function exportToExcel() {
  if (recognitionResults.length === 0) {
    alert("没有可导出的结果");
    return;
  }

  // 显示加载状态
  exportBtn.disabled = true;
  exportBtn.textContent = "导出中...";

  try {
    const result = await invoke<ExcelExportResult>("export_idcard_to_excel", {
      results: recognitionResults,
    });

    if (result.success && result.excel_base64) {
      // 下载 Excel 文件
      downloadExcel(result.excel_base64, `身份证识别结果_${new Date().toLocaleDateString()}.xlsx`);

      // 显示导出完成提示
      alert(`导出完成!\n\n总计: ${result.total} 条\n已写入: ${result.written} 条\n已跳过(字段不足3个): ${result.skipped} 条`);
    } else {
      alert(result.message || "导出失败");
    }
  } catch (error) {
    console.error("导出失败:", error);
    alert("导出失败: " + error);
  } finally {
    exportBtn.disabled = false;
    exportBtn.textContent = "导出Excel";
  }
}

function downloadExcel(base64: string, filename: string) {
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  const blob = new Blob([bytes], { type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// 下载为 ZIP 压缩包
async function downloadAsZip() {
  if (correctionResults.length === 0) {
    alert("没有可下载的图片");
    return;
  }

  const downloadBtn = document.getElementById("downloadZipBtn") as HTMLButtonElement;
  const originalText = downloadBtn.innerHTML;
  downloadBtn.disabled = true;
  downloadBtn.innerHTML = `
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" width="16" height="16" style="margin-right: 6px;">
      <circle cx="12" cy="12" r="10" stroke-dasharray="4 4"></circle>
    </svg>
    ${zipDownloadOptions.auto_rename ? "识别中..." : "打包中..."}
  `;

  try {
    let imageItems: ImageItem[] = [];

    if (zipDownloadOptions.auto_rename) {
      // 如果需要自动重命名，先进行身份证识别
      downloadBtn.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" width="16" height="16" style="margin-right: 6px;">
          <circle cx="12" cy="12" r="10" stroke-dasharray="4 4"></circle>
        </svg>
        正在识别身份证信息...
      `;

      // 使用原始文件进行识别，确保识别结果与身份证识别模式一致
      const imageInputs: ImageInput[] = [];
      for (let i = 0; i < selectedFiles.length; i++) {
        const file = selectedFiles[i];
        const base64 = await fileToBase64(file);
        imageInputs.push({
          filename: file.name,
          image_base64: base64,
        });
      }

      // 调用批量识别
      const batchResult = await invoke<BatchRecognitionResult>("batch_recognize_idcard", {
        images: imageInputs,
      });

      // 使用识别结果构建图片项（包含字段信息），但使用矫正后的图片
      // 建立文件名到矫正结果的映射
      const correctionMap = new Map(correctionResults.map(r => [r.filename, r.corrected_image]));
      
      imageItems = batchResult.results.map((result) => ({
        filename: result.filename,
        image_base64: correctionMap.get(result.filename) || result.corrected_image,
        fields: result.fields,
      }));

      downloadBtn.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" width="16" height="16" style="margin-right: 6px;">
          <circle cx="12" cy="12" r="10" stroke-dasharray="4 4"></circle>
        </svg>
        打包中...
      `;
    } else {
      // 不需要重命名，直接使用矫正结果
      imageItems = correctionResults.map((result) => ({
        filename: result.filename,
        image_base64: result.corrected_image,
        fields: undefined,
      }));
    }

    const result = await invoke<ZipDownloadResult>("download_images_as_zip", {
      images: imageItems,
      options: zipDownloadOptions,
    });

    if (result.success && result.zip_base64) {
      // 下载 ZIP 文件
      downloadZip(result.zip_base64, result.filename);
      alert(`下载成功！\n${result.message}`);
    } else {
      alert(result.message || "下载失败");
    }
  } catch (error) {
    console.error("下载失败:", error);
    alert("下载失败: " + error);
  } finally {
    downloadBtn.disabled = false;
    downloadBtn.innerHTML = originalText;
  }
}

// 下载 ZIP 文件
function downloadZip(base64: string, filename: string) {
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  const blob = new Blob([bytes], { type: "application/zip" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// 初始化应用
init();
