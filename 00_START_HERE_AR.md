# IC-PRO-S — مستودع واحد لGitHub Desktop وZenodo

هذه **نسخة مراجعة قبل النشر العام**. تحتوي على كود IC-PRO-S والبيانات المحفوظة المسموح بإتاحتها والنماذج وملفات إعادة الحساب.

## قبل النشر العام
- راجعي `docs/EXTERNAL_DATA_AND_LICENSE.md`، وحقوق إعادة توزيع الملفات المشتقة والأوزان، وحددي ترخيص الكود المملوك لك. ملف MIT الحالي **مسودة غير مفعّلة**.
- لا تضعي رموز دخول Kaggle أو GitHub أو مفاتيح خدمات داخل المستودع.
- دمج الملفات في ZIP واحد لا يعني أن كل التجارب أُعيد تنفيذ تدريبها تاريخيًا؛ سياسة OOD التشغيلية 12000/600 غير مستعادة بالكامل.

## GitHub Desktop → Zenodo
1. فكّي هذا ZIP؛ الناتج مجلد `IC_PRO_S_REPRODUCIBILITY` واحد.
2. افتحي GitHub Desktop، File > Add Local Repository، اختاري المجلد. إذا ظهرت رسالة Not a Git repository، اختاري Create a repository here.
3. تأكدي في Changes من وجود مجلدات `frozen_inputs` و`checkpoints` وملفات `.pt`.
4. بعد مراجعة الحقوق والترخيص، Commit ثم Publish repository. يجب أن يكون **Public** للاستفادة من Zenodo GitHub Sync التلقائي.
5. في Zenodo، اربطي GitHub بالحساب من Linked accounts، ومن Profile > GitHub اختاري Sync now ثم فعّلي هذا المستودع.
6. **بعد التفعيل** في GitHub افتحي Releases > Draft a new release وأنشئي Tag مثل `v1.0.0`. Zenodo يُنشئ نسخة أرشيفية وDOI عند معالجة Release؛ Commit/Push وحده لا يكفي.
7. تحققي من DOI والملفات المؤرشفة، وضعي DOI الحقيقي في Code Availability وفي خطاب الرد.

## اختبار الملفات
`python scripts/verify_release.py`
`python RUN_REVIEWER_CHECKS.py`
`python RUN_REVIEWER_CHECKS.py --ood-recompute`
قد تحتاجين تثبيت الحزم من requirements.txt؛ إعادة حساب بعض النتائج لا تعني إعادة تدريب كافة الشبكات.

## قيد علمي
النماذج وأغلب سجلات القياس محفوظة، وكاشف Table 8 استُعيدت أربع صفوف من نتائجه؛ ولكن إعادة تشغيل المسار التاريخي المستقل لسياسة OOD ببنك 12,000 عينة و600 معايرة وبيانات التوجيه الدقيقة غير متاحة. لا تصفي هذا المستودع بأنه يعيد كل تجربة تاريخية بتطابق كامل.
