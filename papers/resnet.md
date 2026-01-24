## تمهيد
شكلت هذه البنية نقلة في التعلم العميق، ففي حين كان العمق في الشبكات العصبية لا يتجاوز 10 إلى 15 طبقة ما قبل هذه البنية، أصبح عمق الشبكات العصبية مابعدها يتجاوز الـ 100 إلى 150 طبقة. ما شكل نقلة نوعية في هذا المجال. ففي حين كان على طبقات الشبكة العصبية أن تتعلم كافة التحويل من الدخل إلى الخرج، أصبح كافي أن تتعلم كل مجموعة من الطبقات جزء من ذلك التحول، وبتراكم تلك الطبقات التي تمثل اجزاء التحول المطلوب، نحصل على كافة التحول من صور على دخل الشبكة إلى نوع الكائن الموجود داخل تلك الصور على خرجها. حاول أن تتصور ذلك التحول من صورة مؤلفة من 255x255x3 إلى متجه من 1000 قيمة احدى تلك القيم تمثل صنف الكائن الموجود في تلك الصورة. بذلك يمكننا أن نتصور أن ما تعلمته كل مجموعة من الطبقات هو باقي طرح خرج تلك الطبقة من الدخل، وهو أسهل بكثير من تعلم كافة التحويل.

# تعلم المتبقي العميق للتعرف على الصور

## الخلاصة
تزداد صعوبة تدريب الشبكات العصبية بزيادة عمقها، كما هو معلوم.
Deeper neural networks are more difficult to train.

يقدم الباحثون في هذه الورقة إطار عمل ل**تعلم المتبقي** لتسهيل تدريب الشبكات الأعمق بكثير من تلك المستخدمة سابقاً.
We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously.

أعاد الباحثون صياغة الطبقات لتصبح ك**دالة الباقي** بنسبة إلى مدخلاتها، بعد تعليمها، بدلًا من تعليمها ك**دالة بدون مرجعية**.
We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions.

قدم الباحثون أدلة تجريبية شاملة تُظهر أن شبكات البواقي أسهل في التحسين، ويمكنها تحقيق دقة أعلى مع زيادة العمق بشكل ملحوظ.
We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth

قيّم الباحثون شبكات البواقي على مجموعة بيانات شبكة الصور، بعمق يصل إلى 152 طبقة - أي أعمق بثمانية أضعاف من شبكات **مجموعة الهندسة البصرية** [40] مع الحفاظ على تعقيد أقل.
On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers—8×deeper than VGG nets [40] but still having lower complexity.

حققت مجموعة من شبكات البواقي نسبة خطأ 3.57% على مجموعة اختبار شبكة الصور.
An ensemble of these residual nets achieves 3.57% error on the ImageNet test set.

وقد فازت هذه النتيجة بالمركز الأول في مهمة التصنيف ب**تحدي التعرف البصري واسع النطاق لشبكة الصور لعام 2015**.
This result won the 1st place on the ILSVRC 2015 classification task.

كما قدم الباحثون تحليلًا على مجموعة بيانات **المعهد الكندي للأبحاث المتقدمة بـ 10 أصناف** مع 100 و 1000 طبقة.
We also present analysis on CIFAR-10 with 100 and 1000 layers.


------
لعمق التمثيلات التي تتعلمها طبقات الشبكة العصبية أهمية بالغة في مهام التعرّف البصري.
The depth of representations is of central importance for many visual recognition tasks.

حقق الباحثون تحسّنًا بنسبة 28% على مجموعة بيانات **الكائنات العامة في السياق** للكشف عن الكائنات، بفضل **التمثيلات** العميقة للغاية المقدمة في هذه الورقة.
Solely due to our extremely deep representations, we obtain a 28% relative improvement on the COCO object detection dataset.

شكلت **شبكات البواقي** العميقة أساسًا لمشاركات الباحثين في مسابقتي **الكائنات العامة في السياق لعام 2015** و **التعرف البصري واسع النطاق لعام 2015**، حيث حصد الباحثون أيضاً المراكز الأولى في مهام **كشف الكائنات** في **شبكة الصور**، و**تحديد مواقعها**، والكشف عنها وتجزئتها في مجموعة بيانات **الكائنات العامة في السياق**.
Deep residual nets are foundations of our submissions to ILSVRC & COCO 2015 competitions1, where we also won the 1st places on the tasks of ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation.

## المصطلحات التأسيسية

- تعلم المتبقي: جاء الاسم من الشيء الذي يتم تدريب الشبكة على تعلمه بشكل صريح، هو الباقي يُرمز له بـ F(x)، فبدل أن نطلب منها ان تتعلم كافة التحويل المطلوب من الدخل حتى الخرج والذي يرمز له بـ H(x)، ستتعلم كل طبقة أو مجموعة من الطبقات F(x)=H(x)-x ثم تعيد بناء الخرج المطلوب بجمع الباقي F(x) مع الدخل x لتحصل بذلك على H(x) = F(x) + x، لذلك يسمى تعلم المتبقي.

- دالة الباقي: هي الدالة (يعبر عنها كطبقة أو مجموعة طبقات) التي تمثل الفرق بين الخرج المطلوب والمدخلات، ويرمز لها بـ F(x)، لذلك تكون دالة بمرجعية إلى دخلها.

- دالة بدون مرجعية: هي دالة (يعبر عنها كطبقة أو مجموعة طبقات) تشير إلى تحويل الدخل إلى خرج دون الإشارة الصريحة (مرجعية) إلى المدخلات.

- مجموعة الهندسة البصرية: تشير إلى شبكة VGG المستخدمة في هذه الورقة كأساس لمقارنة العمق والتعقيد.

- تحدي التعرف البصري واسع النطاق لشبكة الصور لعام 2015: هو المسابقة ILSVRC-2015 لقياس أداء نماذج التعرف على الصور بناء على مجموعة بيانات ImageNet شبكة الصور.

- تحدي الكائنات العامة في السياق 2015: هو مسابقة COCO-2015 للكشف عن الكائنات، وتجزئتها، وعنونتها في الصور وذلك باستخدام مجموعة بيانات ’الكائنات العامة في السياق‘.

- مجموعة بيانات المعهد الكندي للأبحاث المتقدمة بـ 10 أصناف: هي CIFAR-10 مجموعة بيانات لتصنيف الصور، استخدمها الباحثون لتقييم شبكات البواقي.

- تمثيل: تتعلم طبقات الشبكة العصبية تمثيلاً لنمط أو مجموعة من الأنماط الموجودة في بيانات التدريب، وكلما زاد عمق الشبكة تعلمت الطبقات الأعمق تمثيلات ذو هرمية أغنى، وتلعب هذه التمثيلات دور الأهم في أداء مهام التصنيف.

- الكائنات العامة في السياق: مجموعة بيانات Microsoft COCO واسعة النطاق والمخصصة لاكتشاف الكائنات وتجزئتها، طورتها مايكروسوفت لتدريب وتقييم أداء الشبكات العصبية العميقة على مهام الرؤية الحاسوبية.

- شبكات البواقي: هي شبكات عصبية عميقة للغاية تستخدم ’تعلم المتبقي‘ مع وصلات مختصرة لتسهيل تحسين النماذج العميقة.
تتعلم كل مجموعة طبقات، المكونة لشبكات البواقي، جزء المتبقي من طرح الدخل من التحويل المطلوب، بدل أن تتعلم كافة التحويل، ثم تضيف إلى ما تعلمته الدخل عبر الوصلة المختصرة. يسمح هذا التصميم بتعميق الشبكات العصبية بشكل كبير دون التأثير سلباً على دقة التدريب.

- كشف الكائنات: Objects Detection عملية تحديد صنف لكل كائن موجود في الصور، بالإضافة إلى تحديد مربع يحيط بتلك الكائنات.

- تجزئة: Segmentation هي تعيين صنف لكل بكسل في الصورة، ما يؤدي إلى تجزئة حدود الكائن بدقة بدلا من رسم مربع يحيطه فقط.

- تحديد الموقع: Localization هو تحديد الكائن الرئيسي ومكان وجوده في الصورة باستخدام مربع للإحاطة به.

- تلاشي التدرج: Vanishing Gradient يشير إلى تضاؤل قيم التدرج بشكل كبير مع الانتشار العكسي القادم من نهاية الشبكة، ما يجعل عملية التدريب صعباً للغاية، ويؤدي إلى بطء شديد في تعليم الطبقات الأولى أو عدم تعلمها على الإطلاق، وهي احدى الصعوبات الرئيسية في تدريب الشبكات العميقة.

- انفجار التدرج: Exploding Gradient يشير إلى تضخم قيم التدرج بشكل مفرط مع الانتشار العكسي القادم من نهاية الشبكة، ما يجعل عملية التدريب غير مستقرة، ويؤدي إلى زعزعة استقرار التدريب ومنع التقارب، وهي مشكلة معروفة في تدريب الشبكات العميقة.

- التقارب: Convergence يشير إلى التقدم باتجاه نجاح عملية التدريب في تقليل الخطأ والوصول إلى حل مستقر أثناء تحسين أوزان الطبقات، أما عدم التقارب فيعني فشل ’المُحسِّن‘ optimizer في إيجاد حل جيد، تأتي هذه الورقة ’بتعلم المتبقي‘ لتسهيل التقارب في الشبكات العميقة.

- التهيئة الطبيعية: Normalized Initialization يشير إلى إعطاء قيم ابتدائية عشوائية لأوزان الطبقات من توزع احتمالي طبيعي، ما يحافظ على استقرار قيم الأوزان بحيث لا تتلاشى أو تنفجر قيم التنشيط والتدرجات، ما يساعد الشبكة العميقة على بدء التقارب، وهي احدى العوامل التي جعلت تدريب الشبكات العميقة ممكناً.

- طبقات التطبيع الوسيطة: Intermediate Normalization Layers طبقة من نوع خاص تضاف بين طبقات الشبكة لتحسين استقرار التدريب ومساعدة الشبكة العميقة على التقارب، حيث تعمل على جعل خرج ’التنشيط‘ الطبقة الوسيطة يتبع للتوزيع الطبيعي.

- التدرج العشوائي: Stochastic Gradient Descent (SGD) طريقة تحسين تستخدم لتدريب الشبكات العصبية عبر ’الانتشار العكسي‘ حيث تُعدل أوزان الطبقات بالاعتماد على دفعات صغيرة من البيانات، يدمج هذا المحسّن مع الانتشار العكسي لتقليل خطأ التدريب.

- الانتشار العكسي: Backpropagation يستخدم لحساب التدرجات وتعديل أوزان الطبقات أثناء تدريب الشبكة، بحيث يبدأ بنشر التدرجات وتعديل أوزان الشبكة من نهايتها إلى بدايتها (عكسياً).

- مشكلة التدهور: Degradation Problem تحدث عندما يؤدي إضافة المزيد من الطبقات إلى الشبكة العصبية إلى زيادة خطأ التدريب، على عكس المتوقع. ولا يعود ذلك إلى ’فرط التخصيص‘ Overfitting الذي يظهر خطأ تدريب منخفض وخطأ اختبار مرتفع، بل إلى صعوبة تحسين عدد كبير من أوزان الطبقات المتراكمة.

- دقة التدريب: تقيس Training Accuracy دقة النموذج على بيانات التدريب وأثناء عملية التدريب. يشير انخفاض دقة التدريب (يعني ارتفاع خطأ التدريب) إلى صعوبة في التحسن.

- طبقة مُطابقة: Identity Mapping Layer تعمل على أن يكون الخرج مساوياً للدخل، بمعنى تمرير الدخل x مباشرة إلى الخرج، ويتم ذلك عملياً من خلال وصلات الاختصار.
## المقدمة

أدت شبكات الطي العصبية العميقة [^22] [^21] إلى سلسلة من الإنجازات في تصنيف الصور [^21] [^49] [^39].
Deep convolutional neural networks [22, 21] have led to a series of breakthroughs for image classification [21, 49, 39]. 

يتكامل طبيعياً وعلى شكل طبقات جسم الشبكات العميقة والذي يمثل الميزات منخفضة ومتوسطة وعالية المستوى [^49] مع رأسها والذي يمثل **المصنف**، وكلما تراكمت الطبقات (زاد عمق الشبكة) زادت تلك المستويات ثراءً.
Deep networks naturally integrate low/mid/highlevel features [49] and classifiers in an end-to-end multilayer fashion, and the “levels” of features can be enriched by the number of stacked layers (depth).

تشير الأدلة حتى نشر هذه الورقة [^40] [^43] إلى أن عمق الشبكة ذو أهمية بالغة، وأن النتائج الرائدة [^40] [^43] [^12] [^16] على مجموعة بيانات صعبة ك**شبكة الصور** [^35] جاءت جميعها من استخدام نماذج "عميقة جدًا" [^40]، بعمق يتراوح بين ستة عشر [^40] وثلاثين [^16].
Recent evidence [40, 43] reveals that network depth is of crucial importance, and the leading results [40, 43, 12, 16] on the challenging ImageNet dataset [35] all exploit “very deep” [40] models, with a depth of sixteen [40] to thirty [16]. 

كما استفادت العديد من مهام التعرف البصري غير البسيطة الأخرى [^7] [^11] [^6] [^32] [^27] بشكل كبير من النماذج العميقة جدًا.
Many other nontrivial visual recognition tasks [7, 11, 6, 32, 27] have also greatly benefited from very deep models.

--------

هنا يبرز السؤال، إنطلاقاً من أهمية العمق: هل تدريب شبكات أفضل بسهولة إضافة المزيد من الطبقات؟ 
Driven by the significance of depth, a question arises: Is learning better networks as easy as stacking more layers? 

كانت العقبات التي تحول دون الإجابة على هذا السؤال هي مشكلتي **تلاشي** و**انفجار** التدرجات المعروفة [^14] [^1] [^8]، والتي تعيق **التقارب** منذ البداية.
An obstacle to answering this question was the notorious problem of vanishing/exploding gradients [14, 1, 8], which hamper convergence from the beginning. 

حُلت هذه المشاكل إلى حد كبير من خلال **التهيئة الطبيعية** [^23] [^8] [^36] [^12] وإضافة **طبقات التطبيع الوسيطة** [^16]، ما مكّن الشبكات ذات العشر طبقات من بدء **التقارب** في خوارزمية **التدرج العشوائي** مع **الانتشار العكسي** [^22].
This problem, however, has been largely addressed by normalized initialization [23, 8, 36, 12] and intermediate normalization layers [16], which enable networks with tens of layers to start converging for stochastic gradient descent (SGD) with backpropagation [22].

------
تظهر **مشكلة تدهور** الأداء، مع بدأ الشبكات العميقة بالتقارب: فمع ازدياد عمق الشبكة، تصل الدقة إلى حدّها الأقصى (وهو أمر متوقع)، ثم تتدهور بسرعة.
When deeper networks are able to start converging, a degradation problem has been exposed: with the network depth increasing, accuracy gets saturated (which might be unsurprising) and then degrades rapidly.

والمثير للدهشة أن هذا التدهور لا ينتج عن **فرط التخصيص**، بل إن إضافة المزيد من الطبقات إلى نموذج عميق مناسب يؤدي إلى زيادة خطأ التدريب، كما ورد في [^10] [^41] وتم التحقق منه بدقة من خلال تجارب الباحثين.
Unexpectedly, such degradation is not caused by overfitting, and adding more layers to a suitably deep model leads to higher training error, as reported in [10, 41] and thoroughly verified by our experiments.

يوضح الشكل 1 مثالًا نموذجيًا.
Fig. 1 shows a typical example.


---
يشير تدهور **دقة التدريب** إلى أن تحسين جميع الأنظمة ليس بنفس السهولة.
The degradation (of training accuracy) indicates that not all systems are similarly easy to optimize. 

لنفرض وجود بنية سطحية، وأخرى أعمق منها أضفنا عليها بعض الطبقات.
Let us consider a shallower architecture and its deeper counterpart that adds more layers onto it.


تُبنى النسخة العميقة بنسخ الطبقات من البنية السطحية وإضافة **طبقات مطابقة** عليها.
There exists a solution by construction to the deeper model: the added layers are identity mapping, and the other layers are copied from the learned shallower model. 

نفهم من هذا التصميم أن النموذج الأعمق يجب أن لا يُعطي خطأ تدريب أكبر من خطأ تدريب البنية السطحية.
The existence of this constructed solution indicates that a deeper model should produce no higher training error than its shallower counterpart. 


أظهرت النتائج أن الخوارزميات الحالية لدينا غير قادرة على إيجاد حلول تتفوق على الحل السابق (أو غير قادرة على ذلك في وقت معقول).
But experiments show that our current solvers on hand are unable to find solutions that are comparably good or better than the constructed solution (or unable to do so in feasible time).

-----------------------------------------

يتناول الباحثون في هذه الورقة **مشكلة التدهور** من خلال تقديم إطار عمل ل**تعلم المتبقي** العميق.
In this paper, we address the degradation problem by introducing a deep residual learning framework.

فبدل أن يأمل الباحثون أن تتعلم كل مجموعة من الطبقات التحويل المطلوب مباشرةً، سمحوا صراحةً لهذه الطبقات بتعلم **دالة الباقي**.
Instead of hoping each few stacked layers directly fit a desired underlying mapping, we explicitly let these layers fit a residual mapping.

يُرمز رياضياً لدالة الربط (التحويل) المطلوب تعلمه بـ H(x)، ويُسمح لمجموعة الطبقات غير الخطية أن تتعلم دالة ربط أخرى هي F(x) := H(x)-x.
Formally, denoting the desired underlying mapping as H(x), we let the stacked nonlinear layers fit another mapping of F(x) := H(x)−x. 

وبذلك، تُعاد صياغة دالة الربط الأصلية إلى F(x)+x.
The original mapping is recast into F(x)+x. 

ويفترض الباحثون أن تحسين **دالة الربط المتبقي** أسهل من تحسين دالة الربط الأصلية **غير المرجعية**.
We hypothesize that it is easier to optimize the residual mapping than to optimize the original, unreferenced mapping.

وفي أقصى الحالات، إذا كانت دالة الربط المطابقة هي الأمثل، فسيكون من الأسهل جعل الباقي يساوي صفرًا من مطابقة دالة الربط المطابقة بواسطة مجموعة من الطبقات غير الخطية.
To the extreme, if an identity mapping were optimal, it would be easier to push the residual to zero than to fit an identity mapping by a stack of nonlinear layers.


## المصدر
https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf


## المراجع
[^1]: Y. Bengio, P. Simard, and P. Frasconi. Learning long-term dependencies with gradient descent is difficult. IEEE Transactions on Neural Networks, 5(2):157–166, 1994.
[^2]: C. M. Bishop. Neural networks for pattern recognition. Oxford university press, 1995.
[^6]: R. Girshick. Fast R-CNN. In ICCV, 2015.
[^7]: R. Girshick, J. Donahue, T. Darrell, and J. Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. In CVPR, 2014.
[^8]: X. Glorot and Y. Bengio. Understanding the difficulty of training deep feedforward neural networks. In AISTATS, 2010.
[^10]: K. He and J. Sun. Convolutional neural networks at constrained time cost. In CVPR, 2015.
[^11]: K. He, X. Zhang, S. Ren, and J. Sun. Spatial pyramid pooling in deep convolutional networks for visual recognition. In ECCV, 2014.
[^12]: K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In ICCV, 2015.
[^14]: S. Hochreiter. Untersuchungen zu dynamischen neuronalen netzen. Diploma thesis, TU Munich, 1991.
[^16]: S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In ICML, 2015.
[^21]: A. Krizhevsky, I. Sutskever, and G. Hinton. Imagenet classification with deep convolutional neural networks. In NIPS, 2012.
[^22]: Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D. Jackel. Backpropagation applied to handwritten zip code recognition. Neural computation, 1989.
[^23]: Y. LeCun, L. Bottou, G. B. Orr, and K.-R.M¨uller. Efficient backprop. In Neural Networks: Tricks of the Trade, pages 9–50. Springer, 1998.
[^27]: J. Long, E. Shelhamer, and T. Darrell. Fully convolutional networks for semantic segmentation. In CVPR, 2015.
[^32]: S. Ren, K. He, R. Girshick, and J. Sun. Faster R-CNN: Towards real-time object detection with region proposal networks. In NIPS, 2015.
[^33]: B. D. Ripley. Pattern recognition and neural networks. Cambridge university press, 1996.
[^35]: O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, et al. ImageNet large scale visual recognition challenge. arXiv:1409.0575, 2014.
[^36]: A. M. Saxe, J. L. McClelland, and S. Ganguli. Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. arXiv:1312.6120, 2013.
[^39]: P. Sermanet, D. Eigen, X. Zhang, M. Mathieu, R. Fergus, and Y. Le-Cun. Overfeat: Integrated recognition, localization and detection using convolutional networks. In ICLR, 2014.
[^40]: K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. In ICLR, 2015.
[^41]: R. K. Srivastava, K. Greff, and J. Schmidhuber. Highway networks. arXiv:1505.00387, 2015.
[^43]: C. Szegedy,W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. In CVPR, 2015.
[^44]: R. Szeliski. Fast surface interpolation using hierarchical basis functions. TPAMI, 1990.
[^48]: W. Venables and B. Ripley. Modern applied statistics with s-plus. 1999.
[^49]: M. D. Zeiler and R. Fergus. Visualizing and understanding convolutional neural networks. In ECCV, 2014.
