# قائمة خطوات مشروع تعلم الألة
سترشدك قائمة الخطوات هذه إلى العمل المنظم خلال مشاريع تعلم الآلة التي تعمل عليها. وهنا الخطوات الثمانية الرئيسية، التي يمكنك اعادة ضبطها وتعديلها لتتناسب واحتياجاتك واحتياجات مشروعك:
1. ضع إطاراً عاماً للمشكلة، ثم إنظر إلى الصورة الكبيرة
2. احصل على البيانات
3. استكشف هذه البيانات واستخلص الرؤى
4. حضر هذه البيانات لعرض أنماطها الأساسية بشكل أفضل لخوارزميات تعلم الآلة
5. جرب عدة نماذج وخوارزميات وأعد قائمة مختصرة للنماذج الأفضل
6. اضبط متغيرات النموذج ثم ادمجه في النظام الكامل
7. اعرض النموذج والنظام
8. أطلق، وراقب النموذج ثم اعمل على صيانته باستمرار


## ضع إطاراً عاماً للمشكلة، ثم إنظر إلى الصورة الكبيرة
حاول أن تسأل الأسئلة التشخيصية لفريق العمل، ثم اجمع الأجوبة وقم بالتحقق منها فذلك ما يجعل منك عالم بيانات متميز

1. تحديد الهدف من النموذج بمصطلحات العمل
2. كيف سيتم استخدام هذا النموذج؟
3. ماهي الحلول الحالية؟ أو الحلول الملتوية المستخدمة حالياً؟
4. كيف يمكن تأطير هذه المشكلة؟ (تعليم بمعلم/تعليم بدون معلم، ...)
5. كيف سيتم قياس أداء النموذج؟
6. هي يتماشى مقياس الأداء مع أهداف العمل؟
7. ماهو الحد الأدنى للدقة المطلوبة اللازمة للوصول إلى أهداف العمل؟
8. ماهي المشاكل المماثلة؟ هي يمكنك إعادة استخدام الخبرات أو الأدوات السابقة؟
9. هل هناك خبير في هذا المجال؟
10. كيف كانت هذه المشكلة تحل يدوياً سابقاً؟
11. ضغ قائمة بكافة الفرضيات او الأجابات التي قمت بها إنت أو فريقك للإجابة على هذه الأسئلة وغيرها
12. حاول التحقق من هذه الفرضيات أو الإجابات إن أمكن

## احصل على البيانات
حاول أتمتة خطواتك بأكبر قدر ممكن حتى تتمكن من الحصول على بيانات جديدة بسهولة

1. ضع قائمة بالبيانات التي تحتاجها والمقدار الذي تحتاجه.
2. أبحث ووثق من أين يمكنك الحصول على هذه البيانات
3. تحقق من توفر المساحة الكافية لاستيعاب هذه البيانات
4. تحقق من القيود القانونية للوصول إلى هذه البيانات، ثم تأكد من حصولك على إذن الوصول إذا لزم الأمر
5. تأكد من حصولك على إذن وصول للبيانات لمدة المشروع
6. قم بإنشاء مساحة عمل جديدة مع مساحة تخزين كافية
7. احصل على البيانات
8. حول البيانات إلى صيغة يمكنك معالجتها بسهولة (دون تغيير البيانات نفسها)
9. تأكد من حذف المعلومات الحساسة أو حمايتها (كأخفاء الهوية مثلاً)
10. تحقق من حجم ونوع البيانات (سلاسل زمنية، عينات، مواقع جغرافية, إلخ)
11. قم بإقتطاع مجموعة الأختبار، وضعها جانباً ولا تنظر إليها أبداً (لا تتطفل عليها)

## استكشف هذه البيانات واستخلص الرؤى
حاول أن تحصل على أراء الخبراء في مجال العمل، ستلعب الدور الأهم في نجاح المشروع

1. أنشئ نسخة من البيانات لأغراض التجربة والاستكشاف (يمكن أن تكون عينة من البيانات إذا كانت ذات حجم كبير، بحيث يمكن التحكم بها بسهولة)
2. أنشئ دفتر ملاحظات Jupyter للاحتفاظ بسجل استكشاف البيانات
3. ادرس الميزات أو الأعمدة وخصائصها
    - الأسم
    - النوع (أصناف أو فئات، رقمي أو عائم، محدد أو غير محدد، نصي، مهيكل أو غير مهيكل، إلخ)
    - نسبة القيم المفقودة في كل ميزة أو عمود
    - الضوضاء وأنواعها (عشوائية، قيم متطرفة، اخطاء تقريب، إلخ)
    - حدد أهمية الميزات أو الأعمدة للمهمة المطلوبة
    - حدد نوع التوزيع الاحتمالي لكل عمود أو ميزة (طبيعي، موحد، لوغارتمي، إلخ)
4. إذا كانت المهمة من نوع التعلم بمعلم فقم بتحديد عمود الهدف أو العنوان
5. اعرض البيانات بشكل مرئي
6. ادرس الترابط بيع اعمدة البيانات الموجودة لديك
7. ادرس كيف يمكن حل هذه المشكلة يدوياً
8. حدد التحويلات التي يمكن تطبيقها على الأعمدة لتساهم في أداء أفضل للمهمة.
9. حدد البيانات الاضافية التي يمكن أن تساهم في أداء أفضل للمهمة
10. وثق كل ما تعلمته

## تحضير البيانات
- اعمل على نسخة جانبية من البيانات، وحافظ على النسخة الأصلية سليمة
- اكتب اجراء منفصل لكل تحويل من تحويلات البيانات التي تقوم بتطبيقها على الأعمدة، وذلك للأسباب التالية
  - حتى تتمكن من تحضير البيانات بسهولة في المرة التالية التي تحصل فيها على نسخة جديدة من البيانات
  - يمكنك تطبيق هذه التحويلات في المشاريع المستقبلية
  - لتحضير مجموعة بيانات الاختبار وتنظيفيها
  - لتحضير وتنظيف البيانات الجديدة عندما ترفع المشروع على البيئة الحية
  - لتسهيل التعامل مع خيارات تحضير البيانات التي قمت بإنشائها على أنها جزء من معاملات خوارزمية التعلم hypyerparameters

1. تنظيف البيانات
   - قم بإصلاح أو إزالة القيم المتطرفة (اختياري)
   - املئ القيمة المفقودة (بصفر، أو المتوسط، أو الوسيط...) أو احذف السطور أو الأعمدة الخاصة بها
2. اختر الميزات أو الأعمدة (اختياري):
   - احذف الميزات التي لا توفر معلومات مهمة للمهمة التي تعمل عليها
3. قم بهندسة الميزات إذا اقتضت الحاجة
   - قطع القيمة المستمرة إلى فئات
   - فكك الميزات أو الأعمدة إلى مركبات ابسط (فئات أو أصناف، التاريخ والوقت، إلخ)
   - اضف بعض التحويلات الواعدة كميزات جديدة (log(x)، sqrt(x)، x^2)
   - جمع بعض الميزات بميزة واعدة جديدة 
4. تحجيم الميزات: مثل التوحيد normalization أو تطبيعها standarization

## حضر قائمة مختصرة للنماذج الواعدة
- قد يكون مفيداً أن تأخذ عينة من مجموعة بيانات التدريب حتى تتمكن من تدريب عدة نماذج مختلفة في وقت معقول هذا إذا كانت البيانات ضخمة (لكن إنتبه أن هذا قد يقلل من دقة النماذج الكبيرة مثل الشبكات العصبية والغابات العشوائية)
- حاول أتمتة هذه الخطوات بأكبر قدر ممكن

1. درّب العديد من النماذج السريعة والصغيرة من فئات مختلفة من الخوارزميات (خطية، Naive Bayes، SVM، غابة عشوائية، شبكة عصبية، إلخ) باستخدام القيمة الافتراضية لبرمترات خوارزميات التعلم hypyerparameters
2. قم بقياس ومقارنة أداء كل نموذج مع الآخر
   - استخدم التحقق المتقاطع Cross-Validation لحساب متوسط الخطأ والإنحراف المعياري على n مرة من عمليات التحقق
3. حلل أهم المتغيرات لكل خوارزمية
4. حلل أنواع الأخطاء التي تحدثها كل خوارزمية على حدى
   - حدد البيانات التي يستعملها الخبير البشري لتجنب مثل هذه الأخطاء
5. قم بجولة سريعة على هندسة وتحديد الميزات المستخدمة
6. كرر الخطوات السابقة من 1 حتى 5 مرة أو مرتين
7. ضع قائمة بأفضل 3 إلى 5 نماذج واعدة، مفضلاً النماذج التي ترتكب أنواعاً مختلفة من الأخطاء

## اضبط متغيرات النظام
- سترغب باستخدام كافة بيانات التدريب في هذه الخطوة، خاصة عندما تتقدم نحو الضبط الدقيق النهائي للنموذج
- حاول أتمتة أكبر قدر ممكن كما هو الحال دائماً

1. اضبط برامترات الخوارزمية Hyperparameters باستخدام التحقق المتقاطع Cross-Validation
    - تعامل مع خيارات تحويل البيانات التي قمت بكتابتها في اجرائات منفصلة كأنها برامترات خوارزمية التعلم Hayperparameters، خاصة عندما لا تكون متأكد منها (على سبيل المثال هي يجب استبدال القيم المفقودة بصفر؟ أم بالمتوسط؟ أو أن تقوم بحذف سطورها بشكل كامل؟)
    - أبدء بالبحث العشوائي عن قيم برامترات خوارزمية التعلم Hyperparameter إذا كان عددهم قليل، لكن ذلك لن يفيد إذا كان عدد هذه البرامترات كبير
2. جرب أساليب المجموعات فغالباً مايؤدي الجمع بين أفضل النماذج لديك أداء أفضل من تشغيلها بشكل فردي
3. قم بقياس خطأ التعميم Generalization Error لأداء النموذج على مجموعة الاختبار فقط عندما تكون واثقاً من نموذجك النهائي.
   - لا تقم بتعديل نموذجك بعد قياس خطأ التعميم، لأن ذلك سيفرط بتخصيص Overfitting نموذجك على مجموعة الاختبار.

## قدم عرض لحلك النهائي
1. وثق كافة الخطوات التي قمت بها
2. أنشاء عرض تقديمي جميل
   - تأكد من عرض الصورة الكبيرة أولاً
3. اشرح كيف يحقق حلك أهداف العمل
4. لا تنسى تقديم النقاط التي أثارت اهتمامك برحلة عملك على المشروع
5. تأكد من عرض النتائج الرئيسية التي توصلت لها من خلال رسومات جميلة وعبارت رنانة يسهل تذكرها

## الإطلاق!
1. جهز الحل الخاص بك للبيئة الحية (وصل المدخلات من البيئة الحية، وقم بكتابة وحدات الاختبار، إلخ)
2. اكتب كود لمراقبة أداء النموذج بشكل مستمر ويرسل تنبيه عند هبوط الأداء تحت حد معين
   - احذر من تدهور الأداء البطيء، لأن النماذج تميل إلى الهرم والقدم من تطور البيانات
   - راقب جودة المدخلات على النموذج، لأنه جهاز استشعار معطل قد يرسل قيماً عشوائية، وفريق آخر تعتمد على خرجه كمدخلات قد يبدأ بإعطاء بيانات قديمة لسببٍ ما
3. أعد تدريب النموذج بشكل منتظم كلما تراكمت لديك بيانات جديدة (أتمت قدر الإمكان)