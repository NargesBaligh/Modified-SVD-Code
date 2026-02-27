# فرمت JPEG. این فرمت یکی از رایج‌ترین فرمت‌های تصویری است که توسط OpenCV پشتیبانی می‌شود
# cv2 برای بارگذاری و پردازش تصاویر(به صورت خاکستری) اپن سی وی می باشد.
import cv2
import pywt
import time
import numpy as np
import matplotlib.pyplot as plt

# بارگذاری تصاویر
image_paths = [fr'C:\Users\Asus\Desktop\New folder\a{i}.jpg' for i in range(1, 10)]
images = [cv2.imread(path, 0) for path in image_paths] # (تصاویر تک کانال) تبدیل به خاکستری(0)

# بررسی وجود تصاویر
for i, img in enumerate(images):
    if img is None:
        raise ValueError(f"image {image_paths[i]} doesn't exist!")

# تغییر اندازه تصاویر
# DWT نمی‌تواند روی چنین ابعادی مستقیماً عمل کند، چون نیاز به تقسیم مساوی دارد.
# (123,123) مقاله
target_size = (124, 124) # به‌طور پیش‌فرض با مقادیر خاکستری (سطح خاکستری متوسط یا نزدیک به 128) پر شوند، مگر اینکه روش دیگری مشخص شود.
images = [cv2.resize(img, target_size, interpolation=cv2.INTER_AREA) for img in images] #روش درون‌یابی برای کاهش کیفیت است که مناسب کاهش اندازه است.

# لیست برای ذخیره تصاویر
originals = [img.copy() for img in images]
# تولید تصاویر کم‌کیفیت با SVD (مثل کد 3)
def create_low_quality_image(image, threshold=0.5):
    # بهتر است محاسبات با float انجام شود
    image_f = image.astype(np.float32)

    U, s, Vt = np.linalg.svd(image_f, full_matrices=False)
    s[s < threshold] = 0  # صفر کردن مقادیر تکین کوچک‌تر از آستانه
    low_quality = U @ np.diag(s) @ Vt

    return np.clip(low_quality, 0, 255).astype(np.uint8)

low_quality_images = [create_low_quality_image(img, threshold=0.5) for img in originals]

svd_images = []
dwt_svd_images = []
swt_images = []
msvd_images = []
######################################################
# تابع‌های معیارهای ارزیابی
def calculate_mse(original, enhanced):
    return np.mean((original - enhanced) ** 2)

def calculate_psnr(original, enhanced):
    mse = calculate_mse(original, enhanced)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 10 * np.log10((max_pixel ** 2) / mse)

def calculate_entropy(image):
    hist = np.histogram(image, bins=256, range=(0, 255))[0]
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def calculate_contrast(image):
    return np.std(image)

def calculate_ssim(original, enhanced):
    mu_x, mu_y = np.mean(original), np.mean(enhanced)
    sigma_x, sigma_y = np.std(original), np.std(enhanced)
    sigma_xy = np.mean((original - mu_x) * (enhanced - mu_y))
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x ** 2 + sigma_y ** 2 + C2))

def calculate_ief(original, low_quality, enhanced):
    mse1 = calculate_mse(original, low_quality)
    mse2 = calculate_mse(original, enhanced)
    return mse1 / mse2 if mse2 != 0 else float('inf')

# محاسبه عدد حالت
def calculate_condition_number(singular_values):
    if len(singular_values) == 0:
        return float('inf')
    sigma_max = np.max(singular_values)
    sigma_min = np.min(singular_values[singular_values > 0])  # جلوگیری از تقسیم بر صفر
    return sigma_max / sigma_min if sigma_min > 0 else float('inf')

def calculate_condition_number_m(s_orig, s_thresholded, s_high, threshold):
    if len(s_orig) == 0 or len(s_thresholded) == 0 or len(s_high) == 0:
        return float('inf')
    sigma_s1 = np.max(s_orig)  # بزرگ‌ترین مقدار تکین اولیه
    # تعداد مقادیر کمتر از آستانه
    p = np.sum(s_thresholded < threshold)
    # کوچک‌ترین مقدار باقی‌مانده پس از حذف p مقدار
    sigma_sn_p = np.sort(s_thresholded[s_thresholded >= threshold])[0] if len(s_thresholded[s_thresholded >= threshold]) > 0 else 1e-10  # مقدار کوچک پیش‌فرض
    # مقدار تکین تصویر با کیفیت بالا برای جایگزینی (متوسط مقادیر بالای آستانه در s_high)
    sigma_sp = np.mean(s_high[s_high >= threshold]) if len(s_high[s_high >= threshold]) > 0 else 1e-10  # مقدار پیش‌فرض
    #print(f"Debug - sigma_s1: {sigma_s1}, sigma_sn_p: {sigma_sn_p}, sigma_sp: {sigma_sp}")  # دیباگ
    denominator = sigma_sn_p + sigma_sp
    return sigma_s1 / denominator if denominator > 0 else float('inf')

def calculate_condition_number_plus(singular_values):
    if len(singular_values) == 0:
        return 0
    sigma_s1 = np.max(singular_values)  # بزرگ‌ترین مقدار تکین
    sigma_sum = np.sum(singular_values)  # مجموع همه مقادیر تکین
    return sigma_s1 / sigma_sum if sigma_sum > 0 else float('inf')


###########################################################################
# پردازش تصاویر
for i in range(len(low_quality_images)):
    img_low = low_quality_images[i]
    img_high = originals[i]

    # SVD
    U, s, Vt = np.linalg.svd(img_low, full_matrices=False)
    s_orig = s.copy()  # ذخیره مقادیر تکین اولیه برای SVD
    U_high, s_high, Vt_high = np.linalg.svd(img_high, full_matrices=False)  # مقادیر تکین تصویر با کیفیت بالا
    threshold = 0.01
    s[s < threshold] = 0
    svd_img = np.dot(U, np.dot(np.diag(s), Vt))
    # اطمینان از اینکه خروجی در محدوده معتبر است و محدود کردن مقادیر به محدوده پیکسل
    svd_img = np.clip(svd_img, 0, 255).astype(np.uint8)
    svd_images.append(svd_img)

    # MSVD
    Ug, sg, Vtg = np.linalg.svd(img_high, full_matrices=False)
    Ub, sb, Vtb = np.linalg.svd(img_low, full_matrices=False)
    threshold = 0.01
    sm = sb.copy()
    sm[sm < threshold] = sg[sm < threshold]  # جایگزینی مقادیر کمتر از آستانه
    sm[sm >= threshold] = sm[sm >= threshold]   # تقویت مقادیر بزرگ‌تر
    enhanced_img = np.dot(Ub, np.dot(np.diag(sm), Vtb))
    enhanced_img = np.clip(enhanced_img, 0, 255).astype(np.uint8)
    msvd_images.append(enhanced_img)

    # DWT-SVD
    coeffs = pywt.dwt2(img_low, 'haar') # تبدیل موجک با فیلتر Haar
    cA, (cH, cV, cD) = coeffs
    U, s, Vt = np.linalg.svd(cA, full_matrices=False) #فقط ماتریس‌های فشرده را محاسبه می‌کند (برای کاهش مصرف حافظه)
    s_orig_dwt = s.copy()  # ذخیره مقادیر تکین اولیه برای DWT-SVD
    U_high_dwt, s_high_dwt, Vt_high_dwt = np.linalg.svd(img_high, full_matrices=False)  # مقادیر تکین تصویر با کیفیت بالا
    threshold = 0.01
    s[s < threshold] = 0
    cA_reconstructed = np.dot(U, np.dot(np.diag(s), Vt))
    coeffs_reconstructed = (cA_reconstructed, (cH, cV, cD))   # بازسازی تصویر با DWT معکوس
    dwt_svd_img = pywt.idwt2(coeffs_reconstructed, 'haar')
    dwt_svd_img = np.clip(dwt_svd_img, 0, 255).astype(np.uint8)
    dwt_svd_images.append(dwt_svd_img)

    # SWT
    coeffs = pywt.swtn(img_low, 'haar', level=1, start_level=0)[0] #فقط یک سطح تجزیه انجام می‌شود و از سطح صفر شروع می‌کند 
    cA = coeffs['aa']                                              # [0]:ضرایب سطح اول را استخراج می‌کند (یک دیکشنری شامل aa, ad, da, dd)
    U, s, Vt = np.linalg.svd(cA, full_matrices=False)
    s_orig_swt = s.copy()  # ذخیره مقادیر تکین اولیه برای SWT
    U_high_swt, s_high_swt, Vt_high_swt = np.linalg.svd(img_high, full_matrices=False)  # مقادیر تکین تصویر با کیفیت بالا
    threshold = 0.01
    s[s < threshold] = 0
    cA_reconstructed = np.dot(U, np.dot(np.diag(s), Vt))
    coeffs['aa'] = cA_reconstructed
    swt_img = pywt.iswtn([coeffs], 'haar')
    swt_img = np.clip(swt_img, 0, 255).astype(np.uint8)
    swt_images.append(swt_img)

#################################################################

    # محاسبه عدد حالت
    k_b_svd = calculate_condition_number(s_orig)  # عدد حالت قبل از آستانه برای SVD
    k_m_svd = calculate_condition_number_m(s_orig, s, s_high, threshold)  # بعد از آستانه با جایگزینی
    k_m_plus_svd = calculate_condition_number_plus(s)  # مجموع مقادیر تکین پس از آستانه
    k_b_msvd = calculate_condition_number(sb)  # قبل از آستانه برای MSVD
    k_m_msvd = calculate_condition_number_m(s_orig, sm, sg, threshold)  # بعد از جایگزینی برای MSVD
    k_m_plus_msvd = calculate_condition_number_plus(sm)  # مجموع مقادیر تکین پس از جایگزینی
    k_b_dwt_svd = calculate_condition_number(s_orig_dwt)  # قبل از آستانه برای DWT-SVD
    k_m_dwt_svd = calculate_condition_number_m(s_orig_dwt, s, s_high_dwt, threshold)  # بعد از آستانه برای DWT-SVD
    k_m_plus_dwt_svd = calculate_condition_number_plus(s)  # مجموع مقادیر تکین پس از آستانه
    k_b_swt = calculate_condition_number(s_orig_swt)  # قبل از آستانه برای SWT
    k_m_swt = calculate_condition_number_m(s_orig_swt, s, s_high_swt, threshold)  # بعد از آستانه برای SWT
    k_m_plus_swt = calculate_condition_number_plus(s)  # مجموع مقادیر تکین پس از آستانه

    # محاسبه معیارها
    mse_svd = calculate_mse(img_high, svd_img)
    psnr_svd = calculate_psnr(img_high, svd_img)
    entropy_svd = calculate_entropy(svd_img)
    contrast_svd = calculate_contrast(svd_img)
    ssim_svd = calculate_ssim(img_high, svd_img)
    ief_svd = calculate_ief(img_high, img_low, svd_img)

    mse_msvd = calculate_mse(img_high, enhanced_img)
    psnr_msvd = calculate_psnr(img_high, enhanced_img)
    entropy_msvd = calculate_entropy(enhanced_img)
    contrast_msvd = calculate_contrast(enhanced_img)
    ssim_msvd = calculate_ssim(img_high, enhanced_img)
    ief_msvd = calculate_ief(img_high, img_low, enhanced_img)

    mse_dwt_svd = calculate_mse(img_high, dwt_svd_img)
    psnr_dwt_svd = calculate_psnr(img_high, dwt_svd_img)
    entropy_dwt_svd = calculate_entropy(dwt_svd_img)
    contrast_dwt_svd = calculate_contrast(dwt_svd_img)
    ssim_dwt_svd = calculate_ssim(img_high, dwt_svd_img)
    ief_dwt_svd = calculate_ief(img_high, img_low, dwt_svd_img)

    mse_swt = calculate_mse(img_high, swt_img)
    psnr_swt = calculate_psnr(img_high, swt_img)
    entropy_swt = calculate_entropy(swt_img)
    contrast_swt = calculate_contrast(swt_img)
    ssim_swt = calculate_ssim(img_high, swt_img)
    ief_swt = calculate_ief(img_high, img_low, swt_img)

    # نمایش معیارها برای هر تصویر
    print(f"Image a{i+1} Metrics:")
    print(f"(SVD) MSE: {mse_svd:.2f}, PSNR: {psnr_svd:.2f}, Entropy: {entropy_svd:.2f}, Contrast: {contrast_svd:.2f}, SSIM: {ssim_svd:.2f}, IEF: {ief_svd:.2f}, K(B): {k_b_svd:.2f}, K(M): {k_m_svd:.2f}, K(M+): {k_m_plus_svd:.2f}")
    print(f"(MSVD) MSE: {mse_msvd:.2f}, PSNR: {psnr_msvd:.2f}, Entropy: {entropy_msvd:.2f}, Contrast: {contrast_msvd:.2f}, SSIM: {ssim_msvd:.2f}, IEF: {ief_msvd:.2f}, K(B): {k_b_msvd:.2f}, K(M): {k_m_msvd:.2f}, K(M+): {k_m_plus_msvd:.2f}")
    print(f"(DWT-SVD) MSE: {mse_dwt_svd:.2f}, PSNR: {psnr_dwt_svd:.2f}, Entropy: {entropy_dwt_svd:.2f}, Contrast: {contrast_dwt_svd:.2f}, SSIM: {ssim_dwt_svd:.2f}, IEF: {ief_dwt_svd:.2f}, K(B): {k_b_dwt_svd:.2f}, K(M): {k_m_dwt_svd:.2f}, K(M+): {k_m_plus_dwt_svd:.2f}")
    print(f"(SWT) MSE: {mse_swt:.2f}, PSNR: {psnr_swt:.2f}, Entropy: {entropy_swt:.2f}, Contrast: {contrast_swt:.2f}, SSIM: {ssim_swt:.2f}, IEF: {ief_swt:.2f}, K(B): {k_b_swt:.2f}, K(M): {k_m_swt:.2f}, K(M+): {k_m_plus_swt:.2f}")


# نمایش تصاویر
fig, axes = plt.subplots(9, 6, figsize=(15, 20))
titles = ['Original', 'Low Quality', 'SVD', 'DWT-SVD', 'SWT', 'MSVD']

for i in range(9):
    axes[i, 0].imshow(originals[i], cmap='gray')
    axes[i, 0].set_title(f'a{i+1}')
    axes[i, 0].axis('off')

    axes[i, 1].imshow(low_quality_images[i], cmap='gray')
    axes[i, 1].set_title(f'a{i+1}_low')
    axes[i, 1].axis('off')

    axes[i, 2].imshow(svd_images[i], cmap='gray')
    axes[i, 2].set_title(f'a{i+1}_svd')
    axes[i, 2].axis('off')

    axes[i, 3].imshow(dwt_svd_images[i], cmap='gray')
    axes[i, 3].set_title(f'a{i+1}_dwtsvd')
    axes[i, 3].axis('off')

    axes[i, 4].imshow(swt_images[i], cmap='gray')
    axes[i, 4].set_title(f'a{i+1}_swt')
    axes[i, 4].axis('off')

    axes[i, 5].imshow(msvd_images[i], cmap='gray')
    axes[i, 5].set_title(f'a{i+1}_msvd')
    axes[i, 5].axis('off')

plt.tight_layout()
plt.savefig('processed_images.png')
plt.show()


# نمایش هیستوگرام‌ها
#تکی های درست
for i in range(9):
    # هیستوگرام تصویر اصلی
    plt.figure(figsize=(6, 4))
    plt.hist(originals[i].ravel(), bins=256, range=(0, 250))
    axes[i, 0].set_title(f'H-a{i+1}')
    axes[i, 0].set_xticks([0, 50, 100, 150, 200, 250])
    axes[i, 0].set_yticks([1, 10, 50, 100, 150])

    axes[i, 1].set_title(f'H-a{i+1}_low')
    axes[i, 1].set_xticks([0, 50, 100, 150, 200, 250])
    axes[i, 1].set_yscale('log')
    axes[i, 1].set_ylim(1, 150)
    axes[i, 1].set_yticks([1, 10, 50, 100, 150])

    axes[i, 2].hist(svd_images[i].ravel(), bins=256, range=(0, 250))#
    axes[i, 2].set_title(f'H-a{i+1}_svd')
    axes[i, 2].set_xticks([0, 50, 100, 150, 200, 250])
    axes[i, 2].set_yscale('log')
    axes[i, 2].set_ylim(1, 150)
    axes[i, 2].set_yticks([1, 10, 50, 100, 150])

    axes[i, 3].hist(dwt_svd_images[i].ravel(), bins=256, range=(0, 250))
    axes[i, 3].set_title(f'H-a{i+1}_dwtsvd')
    axes[i, 3].set_xticks([0, 50, 100, 150, 200, 250])
    axes[i, 3].set_yscale('log')
    axes[i, 3].set_ylim(1, 150)
    axes[i, 3].set_yticks([1, 10, 50, 100, 150])

    axes[i, 4].hist(swt_images[i].ravel(), bins=256, range=(0, 250))
    axes[i, 4].set_title(f'H-a{i+1}_swt')
    axes[i, 4].set_xticks([0, 50, 100, 150, 200, 250])
    axes[i, 4].set_yscale('log')
    axes[i, 4].set_ylim(1, 150)
    axes[i, 4].set_yticks([1, 10, 50, 100, 150])

    axes[i, 5].hist(msvd_images[i].ravel(), bins=256, range=(0, 250))
    axes[i, 5].set_title(f'H-a{i+1}_msvd')
    axes[i, 5].set_xticks([0, 50, 100, 150, 200, 250])
    axes[i, 5].set_yscale('log')
    axes[i, 5].set_ylim(1, 150)
    axes[i, 5].set_yticks([1, 10, 50, 100, 150])

    plt.tight_layout()
    plt.show()



# نمایش هیستوگرام‌ها
fig, axes = plt.subplots(9, 6, figsize=(15, 20))

for i in range(9):
    axes[i, 0].hist(originals[i].ravel(), bins=256, range=(0, 256))
    axes[i, 0].set_title(f'H-a{i+1}')

    axes[i, 1].hist(low_quality_images[i].ravel(), bins=256, range=(0, 256))
    axes[i, 1].set_title(f'H-a{i+1}_low')

    axes[i, 2].hist(svd_images[i].ravel(), bins=256, range=(0, 256))
    axes[i, 2].set_title(f'H-a{i+1}_svd')

    axes[i, 3].hist(dwt_svd_images[i].ravel(), bins=256, range=(0, 256))
    axes[i, 3].set_title(f'H-a{i+1}_dwtsvd')

    axes[i, 4].hist(swt_images[i].ravel(), bins=256, range=(0, 256))
    axes[i, 4].set_title(f'H-a{i+1}_swt')

    axes[i, 5].hist(msvd_images[i].ravel(), bins=256, range=(0, 256))
    axes[i, 5].set_title(f'H-a{i+1}_msvd')

plt.tight_layout()
plt.show()





####################
##################
################
#############
