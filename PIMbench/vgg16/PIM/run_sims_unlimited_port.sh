#!/bin/bash

# ایجاد پوشه نتایج در صورت عدم وجود
mkdir -p edited-padding-results-unlimited-port

# اجرای حلقه برای مقادیر توان ۲ از ۱ تا ۱۲۸
for k in 1 2 4 8 16 32 64 128; do
    # محاسبه تعداد زیرآرایه‌ها (حاصل‌ضرب باید ۸۱۹۲ شود)
    subarray_count=$(( 8192 / k ))
    
    # تنظیم متغیر محیطی
    export ONCNN_PES_PER_CORE=$k
    export ONCNN_TOTAL_PES=4096
    export ONCNN_LOG_DETAILS=0

    
    # تعیین نام فایل خروجی به همراه مسیر پوشه
    output_file="edited-padding-results-unlimited-port/edited-padding-k${k}.txt"
    
    echo "Running simulation for ONCNN_PES_PER_CORE=${k} with subarray count=${subarray_count}..."
    
    # اجرای اسکریپت پایتون و ذخیره خروجی (همراه با لاگ‌های خطا) در فایل متنی
    python3 ../../../app_launcher.py \
        --sim-config /home/erfan/Desktop/oncnn-pim/PIMeval-PIMbench/configs/taco/PIMeval_Fulcrum-${subarray_count}.cfg \
        2 ./vgg16.out > "$output_file" 2>&1
        
    echo "Finished. Results saved to $output_file"
    echo "--------------------------------------------------------"
done