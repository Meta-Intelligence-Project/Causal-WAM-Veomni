/* Cosmos uses the same Vue app and transport as the Lingbot controls. */
window.useCosmos = function ({ref, reactive, computed, onMounted, onUnmounted, api, showToast}) {
    const inferenceModel = ref('lingbot');
    const cosmosConfig = reactive({checkpoint: '', server: {}, client: {save_prediction_video: false}, prompt: ''});
    const cosmosState = ref({server: {}, client: {}, checkpoints: [], logs: {}});
    const cosmosBusy = ref(false), cosmosLoaded = ref(false), cosmosError = ref('');
    const cosmosPresets = ref([]), cosmosPresetIndex = ref(''), cosmosSaved = ref('');
    const cosmosLogTab = ref('client');
    let timer, disposed = false, loading = false;
    const cosmosDirty = computed(() => JSON.stringify(cosmosConfig) !== cosmosSaved.value);
    const cosmosGeometry = computed(() => cosmosState.value.server.geometry || ({
        ...(cosmosState.value.checkpoints.find(x => x.relative === cosmosConfig.checkpoint)?.geometry || {}),
        rtc_enabled: cosmosConfig.server.rtc_enabled, rtc_delay: cosmosConfig.server.rtc_delay
    }));
    const cosmosRtcError = computed(() => cosmosGeometry.value.rtc_enabled && (
        cosmosConfig.client.smooth_method !== 'none' || cosmosConfig.client.control_mode === 'eef' ||
        cosmosConfig.client.save_prediction_video
    ) ? 'RTC 要求轨迹平滑为 none、使用关节控制，并关闭预测视频保存。也可关闭 RTC 后重启服务。' : '');
    const cosmosPhase = computed(() => ({starting: '连接设备中', inferring: '模型推理中', executing: '执行动作中', updating_cache: '更新上下文', saving_video: '保存生成视频中', stopped: '已停止', complete: '已完成', failed: '失败'}[cosmosState.value.client.phase] || '未启动'));
    const cosmosLogText = computed(() => (cosmosState.value.logs[cosmosLogTab.value] || []).join('\n') || '暂无日志');
    const cosmosServerFields = [['steps', '去噪步数', 1, 100, 1], ['guidance', '引导强度', 1, 20, .1], ['shift', 'SNR shift', .1, 20, .1], ['history', '历史 latent 帧数', 1, 128, 1], ['seed', '随机种子', 0, 2147483647, 1]];
    async function loadCosmosStatus() {
        if (loading || disposed) return;
        loading = true;
        try {
            const value = await api('/api/cosmos/status');
            if (disposed) return;
            if (value?.ok) { cosmosState.value = value; cosmosError.value = ''; }
            else cosmosError.value = value?.error || '无法读取 Cosmos 状态';
        } catch (_) { if (!disposed) cosmosError.value = '无法连接 Cosmos 管理服务'; }
        finally { loading = false; }
    }
    async function saveCosmosConfig() {
        cosmosBusy.value = true;
        try {
            const value = await api('/api/cosmos/config', {method: 'POST', body: JSON.stringify(cosmosConfig)});
            if (!value?.ok) { showToast(value?.error || 'Cosmos 配置保存失败', 'error'); return false; }
            Object.assign(cosmosConfig, value.config);
            cosmosSaved.value = JSON.stringify(cosmosConfig);
            showToast(value.message);
            return true;
        } catch (_) { showToast('Cosmos 配置保存失败', 'error'); return false; }
        finally { cosmosBusy.value = false; }
    }
    async function cosmosAction(target, action) {
        if (cosmosBusy.value) return;
        if (action !== 'stop' && cosmosDirty.value && !await saveCosmosConfig()) return;
        cosmosBusy.value = true;
        try {
            const value = await api(`/api/cosmos/${target}/${action}`, {method: 'POST'});
            showToast(value?.message || value?.error || 'Cosmos 请求失败', value?.ok ? 'success' : 'error');
            cosmosLogTab.value = target;
            await loadCosmosStatus();
        } catch (_) { showToast('Cosmos 请求失败，请检查连接', 'error'); }
        finally { cosmosBusy.value = false; }
    }
    function selectCosmosPreset(event) {
        const preset = cosmosPresets.value[Number(event.target.value)];
        if (preset) cosmosConfig.prompt = preset.prompt;
    }
    onMounted(async () => {
        try {
            const value = await api('/api/cosmos/config');
            if (value?.ok && !disposed) {
                Object.assign(cosmosConfig, value.config);
                cosmosPresets.value = value.presets;
                cosmosState.value.checkpoints = value.checkpoints;
                cosmosSaved.value = JSON.stringify(cosmosConfig);
                cosmosLoaded.value = true;
            } else if (!disposed) cosmosError.value = value?.error || 'Cosmos 配置读取失败';
        } catch (_) { if (!disposed) cosmosError.value = 'Cosmos 配置读取失败'; }
        await loadCosmosStatus();
        if (!disposed) timer = setInterval(loadCosmosStatus, 1500);
    });
    onUnmounted(() => { disposed = true; clearInterval(timer); });
    return {inferenceModel, cosmosConfig, cosmosState, cosmosBusy, cosmosLoaded, cosmosError, cosmosPresets,
        cosmosPresetIndex, cosmosLogTab, cosmosDirty, cosmosGeometry, cosmosPhase, cosmosLogText,
        cosmosServerFields, cosmosRtcError, loadCosmosStatus, saveCosmosConfig, cosmosAction, selectCosmosPreset};
};
