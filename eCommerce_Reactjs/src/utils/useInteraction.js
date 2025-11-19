import { useCallback } from 'react';
import { logInteraction } from '../services/interactionService';

/**
 * Custom hook để ghi lại user interactions
 * Tự động detect device type và handle errors
 */
const useInteraction = () => {

    /**
     * Phát hiện loại thiết bị
     */
    const detectDevice = useCallback(() => {
        const userAgent = navigator.userAgent.toLowerCase();
        const width = window.innerWidth;

        if (/(tablet|ipad|playbook|silk)|(android(?!.*mobi))/i.test(userAgent)) {
            return 'tablet';
        }
        if (/Mobile|Android|iP(hone|od)|IEMobile|BlackBerry|Kindle|Silk-Accelerated|(hpw|web)OS|Opera M(obi|ini)/.test(userAgent)) {
            return 'mobile';
        }
        // Fallback dựa trên screen width
        if (width < 768) {
            return 'mobile';
        } else if (width >= 768 && width < 1024) {
            return 'tablet';
        }
        return 'desktop';
    }, []);

    /**
     * Ghi lại interaction
     * @param {number} userId - ID của user
     * @param {number} productId - ID của product
     * @param {string} actionCode - 'view' | 'cart' | 'purchase'
     * @param {string} customDevice - optional: override auto-detected device
     */
    const trackInteraction = useCallback(async (userId, productId, actionCode, customDevice = null) => {
        // Kiểm tra input
        if (!userId || !productId || !actionCode) {
            console.warn('trackInteraction: Missing required parameters', { userId, productId, actionCode });
            return { success: false, message: 'Missing required parameters' };
        }

        const device = customDevice || detectDevice();

        try {
            const response = await logInteraction({
                userId,
                productId,
                actionCode,
                device
            });

            if (response?.success) {
                console.log(`✓ Interaction logged: ${actionCode} on product ${productId}`);
                return { success: true, data: response.data };
            } else {
                console.warn('Failed to log interaction:', response?.message);
                return { success: false, message: response?.message };
            }
        } catch (error) {
            console.error('Error logging interaction:', error);
            return {
                success: false,
                message: error.response?.data?.message || error.message
            };
        }
    }, [detectDevice]);

    /**
     * Track view action
     */
    const trackView = useCallback((userId, productId) => {
        return trackInteraction(userId, productId, 'view');
    }, [trackInteraction]);

    /**
     * Track cart action
     */
    const trackCart = useCallback((userId, productId) => {
        return trackInteraction(userId, productId, 'cart');
    }, [trackInteraction]);

    /**
     * Track purchase action
     */
    const trackPurchase = useCallback((userId, productId) => {
        return trackInteraction(userId, productId, 'purchase');
    }, [trackInteraction]);

    return {
        trackInteraction,
        trackView,
        trackCart,
        trackPurchase,
        detectDevice
    };
};

export default useInteraction;
