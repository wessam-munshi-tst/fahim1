/**
 * Admin Panel JavaScript - School Robot
 * @version 4.0.0 - Lessons + phrase manager (multi-tenant)
 */

'use strict';

// Tenant ID is injected by the server for tenant pages (/{tenant}/admin)
const TENANT_ID = window.TENANT_ID || null;
const API_PREFIX = TENANT_ID ? `/${TENANT_ID}` : '';
const DRAFT_KEY = `robotLessonDraft:${TENANT_ID || 'main'}`;

const CONFIG = Object.freeze({
    STATUS_REFRESH_INTERVAL_MS: 3000,
    TOAST_DURATION_MS: 3200,
});

const MODE_LABELS = Object.freeze({
    explain: 'شرح للمادة',
    questions: 'أسئلة',
    story: 'قصة وحوار',
    welcome: 'ترحيب وتفاعل',
});

const $ = (id) => document.getElementById(id);

const el = {
    // stats & quick controls
    connectionCount: $('connectionCount'),
    uptime: $('uptime'),
    killBtn: $('killBtn'),
    muteBtn: $('muteBtn'),
    toast: $('toast'),
    // running lesson
    lessonLive: $('lessonLive'),
    lessonLiveLabel: $('lessonLiveLabel'),
    lessonLiveTime: $('lessonLiveTime'),
    lessonLiveTitle: $('lessonLiveTitle'),
    lessonLiveProgress: $('lessonLiveProgress'),
    lessonStopBtn: $('lessonStopBtn'),
    // lesson builder
    lessonMode: $('lessonMode'),
    lessonForm: $('lessonForm'),
    lessonSubject: $('lessonSubject'),
    lessonTitle: $('lessonTitle'),
    durationGroup: document.querySelector('.segmented'),
    lessonOkBtn: $('lessonOkBtn'),
    lessonGenHint: $('lessonGenHint'),
    lessonResult: $('lessonResult'),
    lessonResultTitle: $('lessonResultTitle'),
    lessonPartsCount: $('lessonPartsCount'),
    lessonContent: $('lessonContent'),
    lessonStartBtn: $('lessonStartBtn'),
    lessonRegenBtn: $('lessonRegenBtn'),
    // voice
    voiceSelect: $('voiceSelect'),
    currentVoice: $('currentVoice'),
    // phrases
    phraseSelect: $('phraseSelect'),
    speakPhraseBtn: $('speakPhraseBtn'),
    managePhrasesBtn: $('managePhrasesBtn'),
    phrasesDialog: $('phrasesDialog'),
    phrasesCloseBtn: $('phrasesCloseBtn'),
    newPhrase: $('newPhrase'),
    addPhraseBtn: $('addPhraseBtn'),
    phraseList: $('phraseList'),
    phraseCount: $('phraseCount'),
    savePhrasesBtn: $('savePhrasesBtn'),
    cancelPhrasesBtn: $('cancelPhrasesBtn'),
    // free text & instructions
    speakText: $('speakText'),
    speakBtn: $('speakBtn'),
    instructionsText: $('instructionsText'),
    instructionsBtn: $('instructionsBtn'),
};

const state = {
    phrases: [],
    phrasesDraft: [],
    lesson: null,            // running lesson info from server
    lessonSyncedAt: 0,       // when lesson.remaining_seconds was received
    generatedFor: null,      // {mode, subject, lesson, duration} of current content
};

// =============================================================================
// Helpers
// =============================================================================

let toastTimer = null;
function showToast(message, isError = false) {
    el.toast.textContent = message;
    el.toast.className = 'toast show ' + (isError ? 'error' : 'success');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.toast.className = "toast"; }, isError && message.length > 60 ? 9000 : CONFIG.TOAST_DURATION_MS);
}

function formatUptime(seconds) {
    if (seconds < 60) return `${Math.floor(seconds)}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${mins}m`;
}

function formatClock(seconds) {
    const s = Math.max(0, Math.round(seconds));
    return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
}

function countParts(text) {
    return text.split(/^\s*-{3,}\s*$/m).map(t => t.trim()).filter(Boolean).length;
}

function partsLabel(n) {
    if (n === 1) return 'جزء واحد';
    if (n === 2) return 'جزآن';
    if (n >= 3 && n <= 10) return `${n} أجزاء`;
    return `${n} جزءًا`;
}

/** Disable a button while an async action runs. */
async function withBusy(button, fn) {
    if (button.disabled) return;
    button.disabled = true;
    button.classList.add('loading');
    try {
        return await fn();
    } finally {
        button.disabled = false;
        button.classList.remove('loading');
    }
}

/** JSON request helper. Returns parsed body, or null on failure (toast shown). */
async function api(path, body = null, method = 'POST') {
    try {
        const options = { method, headers: { 'Content-Type': 'application/json' } };
        if (body !== null) options.body = JSON.stringify(body);
        const response = await fetch(`${API_PREFIX}${path}`, options);
        if (!response.ok) {
            showToast(`خطأ من الخادم: ${response.status}`, true);
            return null;
        }
        const data = await response.json();
        if (data && data.status === 'error') {
            showToast(data.message || 'حدث خطأ', true);
            return null;
        }
        return data;
    } catch (error) {
        showToast('خطأ في الاتصال بالخادم', true);
        return null;
    }
}

function saveDraft() {
    try {
        localStorage.setItem(DRAFT_KEY, JSON.stringify({
            mode: el.lessonMode.value,
            subject: el.lessonSubject.value,
            lesson: el.lessonTitle.value,
            duration: selectedDuration(),
            content: el.lessonContent.value,
            generatedFor: state.generatedFor,
        }));
    } catch (e) { /* storage unavailable */ }
}

function restoreDraft() {
    let draft = null;
    try { draft = JSON.parse(localStorage.getItem(DRAFT_KEY) || 'null'); } catch (e) { draft = null; }
    if (!draft || !draft.mode || !MODE_LABELS[draft.mode]) return;
    el.lessonMode.value = draft.mode;
    el.lessonForm.hidden = false;
    el.lessonSubject.value = draft.subject || '';
    el.lessonTitle.value = draft.lesson || '';
    const radio = document.querySelector(`input[name="lessonDuration"][value="${draft.duration}"]`);
    if (radio) radio.checked = true;
    if (draft.content && draft.generatedFor) {
        state.generatedFor = draft.generatedFor;
        showResult(draft.content);
    }
}

// =============================================================================
// Status / quick controls
// =============================================================================

function updateMuteButton(isMuted) {
    const btnText = el.muteBtn.querySelector('.btn-text');
    btnText.textContent = isMuted ? 'تشغيل الصوت' : 'كتم الصوت';
    el.muteBtn.classList.toggle('muted', isMuted);
}

function applyStatus(status) {
    el.connectionCount.textContent = status.active_connections;
    el.uptime.textContent = formatUptime(status.uptime_seconds);
    el.currentVoice.textContent = status.current_voice;
    updateMuteButton(status.muted);
    state.lesson = status.lesson || null;
    state.lessonSyncedAt = Date.now();
    renderLessonLive();
}

function renderLessonLive() {
    const lesson = state.lesson;
    if (!lesson) {
        el.lessonLive.hidden = true;
        return;
    }
    const elapsedSinceSync = (Date.now() - state.lessonSyncedAt) / 1000;
    const remaining = Math.max(0, lesson.remaining_seconds - elapsedSinceSync);
    const total = lesson.duration * 60;
    const ended = remaining <= 0;

    el.lessonLive.hidden = false;
    el.lessonLive.classList.toggle('ended', ended);
    el.lessonLiveLabel.textContent = ended ? `انتهى وقت الحصة · ${lesson.label}` : `حصة جارية · ${lesson.label}`;
    el.lessonLiveTime.textContent = formatClock(remaining);
    el.lessonLiveTitle.textContent = `${lesson.subject} — ${lesson.lesson} (${partsLabel(lesson.parts)})`;
    el.lessonLiveProgress.style.width = `${Math.min(100, ((total - remaining) / total) * 100)}%`;
}

async function checkStatus() {
    try {
        const response = await fetch(`${API_PREFIX}/admin/status`);
        if (response.ok) applyStatus(await response.json());
    } catch (error) { /* silent */ }
}

async function killAudio() {
    await withBusy(el.killBtn, async () => {
        const res = await api('/admin/kill');
        if (res) { showToast('تم إيقاف جميع الجلسات'); checkStatus(); }
    });
}

async function toggleMute() {
    await withBusy(el.muteBtn, async () => {
        const res = await api('/admin/mute');
        if (res) applyStatus(res);
    });
}

// =============================================================================
// Lesson builder
// =============================================================================

function selectedDuration() {
    const checked = document.querySelector('input[name="lessonDuration"]:checked');
    return checked ? Number(checked.value) : null;
}

function readLessonForm() {
    const req = {
        mode: el.lessonMode.value,
        subject: el.lessonSubject.value.trim(),
        lesson: el.lessonTitle.value.trim(),
        duration: selectedDuration(),
    };
    el.lessonSubject.classList.toggle('invalid', !req.subject);
    el.lessonTitle.classList.toggle('invalid', !req.lesson);
    el.durationGroup.classList.toggle('invalid', !req.duration);

    if (!req.mode) { showToast('اختر نوع النشاط أولًا', true); return null; }
    if (!req.subject) { showToast('اكتب اسم المادة الدراسية', true); el.lessonSubject.focus(); return null; }
    if (!req.lesson) { showToast('اكتب اسم الدرس', true); el.lessonTitle.focus(); return null; }
    if (!req.duration) { showToast('اختر مدة حديث الروبوت', true); return null; }
    return req;
}

function onModeChange() {
    el.lessonForm.hidden = !el.lessonMode.value;
    if (state.generatedFor && state.generatedFor.mode !== el.lessonMode.value) {
        el.lessonResult.hidden = true;
        state.generatedFor = null;
    }
    if (!el.lessonSubject.value) el.lessonSubject.focus({ preventScroll: true });
    saveDraft();
}

function showResult(content) {
    const meta = state.generatedFor;
    el.lessonContent.value = content;
    el.lessonResultTitle.textContent = meta
        ? `${MODE_LABELS[meta.mode]}: ${meta.lesson}`
        : 'المحتوى المولَّد';
    updatePartsCount();
    el.lessonResult.hidden = false;
}

function updatePartsCount() {
    el.lessonPartsCount.textContent = partsLabel(countParts(el.lessonContent.value));
}

async function generateLesson(button) {
    const req = readLessonForm();
    if (!req) return;
    if (!el.lessonResult.hidden && el.lessonContent.value.trim() && button === el.lessonRegenBtn
        && !confirmReplace()) return;

    el.lessonGenHint.hidden = false;
    el.lessonOkBtn.querySelector('.btn-label').textContent = 'جاري التوليد...';
    el.lessonRegenBtn.disabled = true;
    el.lessonOkBtn.classList.add('loading');
    el.lessonOkBtn.disabled = true;
    try {
        const res = await api('/admin/lesson/generate', req);
        if (res) {
            state.generatedFor = req;
            showResult(res.content);
            saveDraft();
            showToast('تم توليد المحتوى، راجعه ثم ابدأ الحصة');
            el.lessonResult.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    } finally {
        el.lessonGenHint.hidden = true;
        el.lessonOkBtn.querySelector('.btn-label').textContent = 'موافق';
        el.lessonOkBtn.classList.remove('loading');
        el.lessonOkBtn.disabled = false;
        el.lessonRegenBtn.disabled = false;
    }
}

function confirmReplace() {
    // window.confirm keeps it simple and works everywhere on mobile.
    return window.confirm('سيُستبدل المحتوى الحالي بمحتوى جديد. هل تريد المتابعة؟');
}

async function startLesson() {
    const content = el.lessonContent.value.trim();
    const meta = state.generatedFor;
    if (!meta) { showToast('ولّد المحتوى أولًا بالضغط على موافق', true); return; }
    if (!content) { showToast('محتوى الحصة فارغ', true); return; }
    await withBusy(el.lessonStartBtn, async () => {
        const res = await api('/admin/lesson/start', { ...meta, content });
        if (res) {
            showToast(res.message || 'بدأت الحصة');
            checkStatus();
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }
    });
}

async function stopLesson() {
    await withBusy(el.lessonStopBtn, async () => {
        const res = await api('/admin/lesson/stop');
        if (res) { showToast(res.message || 'تم إنهاء الحصة'); checkStatus(); }
    });
}

// =============================================================================
// Voice
// =============================================================================

async function loadVoices() {
    try {
        const response = await fetch(`${API_PREFIX}/admin/voices`);
        if (!response.ok) return;
        const data = await response.json();
        el.voiceSelect.replaceChildren(...data.voices.map(v => {
            const opt = new Option(`${v.name} (${v.gender === 'Female' ? 'أنثى' : 'ذكر'})`, v.name);
            opt.selected = v.name === data.current;
            return opt;
        }));
    } catch (error) { /* silent */ }
}

async function changeVoice() {
    const voice = el.voiceSelect.value;
    if (!voice) return;
    el.voiceSelect.disabled = true;
    const res = await api('/admin/voice', { voice });
    el.voiceSelect.disabled = false;
    if (res) { showToast(`تم تغيير الصوت إلى ${voice}`); el.currentVoice.textContent = voice; }
}

// =============================================================================
// Phrases (dropdown + manager window)
// =============================================================================

function renderPhraseSelect() {
    if (!state.phrases.length) {
        el.phraseSelect.replaceChildren(new Option('لا توجد عبارات — أضف من «إدارة العبارات»', ''));
        return;
    }
    const previous = el.phraseSelect.value;
    el.phraseSelect.replaceChildren(...state.phrases.map(p => new Option(p, p)));
    if (state.phrases.includes(previous)) el.phraseSelect.value = previous;
}

async function loadPhrases() {
    try {
        const response = await fetch(`${API_PREFIX}/admin/phrases`);
        if (!response.ok) return;
        const data = await response.json();
        state.phrases = data.phrases || [];
        renderPhraseSelect();
    } catch (error) { /* silent */ }
}

async function speak(text, button) {
    await withBusy(button, async () => {
        const res = await api('/admin/speak', { text });
        if (res) showToast('تم إرسال العبارة للروبوت ✓');
    });
}

function renderPhraseList() {
    const items = state.phrasesDraft.map((phrase, index) => {
        const li = document.createElement('li');
        const span = document.createElement('span');
        span.className = 'phrase-text';
        span.textContent = phrase;
        const del = document.createElement('button');
        del.type = 'button';
        del.className = 'icon-btn danger';
        del.setAttribute('aria-label', 'حذف العبارة');
        del.textContent = '🗑';
        del.addEventListener('click', () => {
            state.phrasesDraft.splice(index, 1);
            renderPhraseList();
        });
        li.append(span, del);
        return li;
    });
    if (!items.length) {
        const li = document.createElement('li');
        li.className = 'empty';
        li.textContent = 'القائمة فارغة';
        items.push(li);
    }
    el.phraseList.replaceChildren(...items);
    el.phraseCount.textContent = state.phrasesDraft.length;
}

function openPhrasesDialog() {
    state.phrasesDraft = [...state.phrases];
    el.newPhrase.value = '';
    renderPhraseList();
    if (typeof el.phrasesDialog.showModal === 'function') {
        el.phrasesDialog.showModal();
    } else {
        el.phrasesDialog.setAttribute('open', '');
    }
    el.newPhrase.focus({ preventScroll: true });
}

function closePhrasesDialog() {
    if (typeof el.phrasesDialog.close === 'function') el.phrasesDialog.close();
    else el.phrasesDialog.removeAttribute('open');
}

function addPhrase() {
    const text = el.newPhrase.value.replace(/\s+/g, ' ').trim();
    if (!text) { showToast('اكتب العبارة أولًا', true); el.newPhrase.focus(); return; }
    if (state.phrasesDraft.includes(text)) { showToast('العبارة موجودة مسبقًا', true); return; }
    state.phrasesDraft.push(text);
    el.newPhrase.value = '';
    renderPhraseList();
    el.phraseList.lastElementChild?.scrollIntoView({ block: 'nearest' });
    el.newPhrase.focus({ preventScroll: true });
}

async function savePhrases() {
    // Include a phrase that was typed but not yet added.
    const pending = el.newPhrase.value.replace(/\s+/g, ' ').trim();
    if (pending && !state.phrasesDraft.includes(pending)) state.phrasesDraft.push(pending);

    await withBusy(el.savePhrasesBtn, async () => {
        const res = await api('/admin/phrases', { phrases: state.phrasesDraft }, 'PUT');
        if (res) {
            state.phrases = res.phrases;
            renderPhraseSelect();
            if (pending) el.phraseSelect.value = pending;
            closePhrasesDialog();
            showToast('تم حفظ العبارات');
        }
    });
}

// =============================================================================
// Free text & instructions
// =============================================================================

async function sendText() {
    const text = el.speakText.value.trim();
    if (!text) { showToast('الرجاء إدخال نص', true); return; }
    await withBusy(el.speakBtn, async () => {
        const res = await api('/admin/speak', { text });
        if (res) { el.speakText.value = ''; showToast('تم إرسال النص'); }
    });
}

async function loadInstructions() {
    try {
        const response = await fetch(`${API_PREFIX}/admin/instructions`);
        if (response.ok) el.instructionsText.value = (await response.json()).instructions;
    } catch (error) { /* silent */ }
}

async function saveInstructions() {
    const instructions = el.instructionsText.value.trim();
    if (!instructions) { showToast('الرجاء إدخال التعليمات', true); return; }
    await withBusy(el.instructionsBtn, async () => {
        const res = await api('/admin/instructions', { instructions });
        if (res) showToast('تم حفظ التعليمات');
    });
}

// =============================================================================
// Wiring
// =============================================================================

el.killBtn.addEventListener('click', killAudio);
el.muteBtn.addEventListener('click', toggleMute);
el.lessonStopBtn.addEventListener('click', stopLesson);

el.lessonMode.addEventListener('change', onModeChange);
el.lessonOkBtn.addEventListener('click', () => generateLesson(el.lessonOkBtn));
el.lessonRegenBtn.addEventListener('click', () => generateLesson(el.lessonRegenBtn));
el.lessonStartBtn.addEventListener('click', startLesson);
el.lessonContent.addEventListener('input', () => { updatePartsCount(); saveDraft(); });
[el.lessonSubject, el.lessonTitle].forEach(input => {
    input.addEventListener('input', () => { input.classList.remove('invalid'); saveDraft(); });
    input.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); el.lessonOkBtn.click(); } });
});
document.querySelectorAll('input[name="lessonDuration"]').forEach(r =>
    r.addEventListener('change', () => { el.durationGroup.classList.remove('invalid'); saveDraft(); }));

el.voiceSelect.addEventListener('change', changeVoice);

el.speakPhraseBtn.addEventListener('click', () => {
    const text = el.phraseSelect.value;
    if (!text) { showToast('لا توجد عبارة مختارة', true); return; }
    speak(text, el.speakPhraseBtn);
});
el.managePhrasesBtn.addEventListener('click', openPhrasesDialog);
el.addPhraseBtn.addEventListener('click', addPhrase);
el.newPhrase.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); addPhrase(); }
});
el.savePhrasesBtn.addEventListener('click', savePhrases);
el.cancelPhrasesBtn.addEventListener('click', closePhrasesDialog);
el.phrasesCloseBtn.addEventListener('click', closePhrasesDialog);
el.phrasesDialog.addEventListener('click', (e) => {
    if (e.target === el.phrasesDialog) closePhrasesDialog(); // tap on backdrop
});

el.speakBtn.addEventListener('click', sendText);
el.instructionsBtn.addEventListener('click', saveInstructions);

restoreDraft();
checkStatus();
loadVoices();
loadPhrases();
loadInstructions();
setInterval(checkStatus, CONFIG.STATUS_REFRESH_INTERVAL_MS);
setInterval(renderLessonLive, 1000);
