- شبكة الصور: جاء الاسم شبكة الصور ImageNet من أن قاعدة بيانات الصور المقدمة في هذه الورقة قد تم تصنيفها بناء على الهيكل الشبكي المقدم في قاعدة بيانات ’شبكة المفردات‘ WordNet والتي تقدم علاقات مختلفة بين كل زوج من مفرداتها مثل المرادفات، والأضداد، والمعاني الخاصة، والمعاني العامة، والمتشابهات وهكذا. قدمت شبكة المفردات مفهوماً رئيسياً يدعى ’مجموعة المرادفات‘ كمجموعة من الكلمات التي تربطها العلاقات السابقة وتتشارك نفس المفهوم.

- شبكة الكلمات: معجم WordNet هو قاعدة بيانات للغة الإنكليزية، تُصنّف الكلمات التي ترتبط فيما بينها بعلاقات دلالية في مجموعات من المعاني المترادفة، تسمى ’مجموعة المرادفات‘. وتوفر هرمية دلالية للمفاهيم التي تستخدمها شبكة الصور لترتيب صورها.

- هيكل مفاهيم: تعرف Ontology كتنظيم بنيوي هرمي للمفاهيم العلاقات الدلالية فيما بينها (مثل، الترادف، الأضداد، إلخ) وتستخدم لتنظيم المعرفة. 

- مجموعة المرادفات: هي Synonymous Set مجموعة من الكلمات التي ترتبط بعلاقات دلالية وتتشارك مفهوماً ذو معنى واحداً في ’شبكة الكلمات‘. تربط ’شبكة الصور‘ كل مجموعة معرفة في ’شبكة الكلمات‘ بالعديد من الصور التي تحتوي ذلك المفهوم.

- شجرة فرعية: هو فرع من الهرمية الدلالية المقدمة في ’شبكة الكلمات‘ (مثل، الثديات) وتحتوي تحتها كل ’مجموعات المرادفات‘ التي تنتمي إليها.


# شبكة الصور الهرمية واسعة النطاق


## الخلاصة


قدم الانفجار الهائل في عدد الصور المتاح على الإنترنت إمكانية تطوير نماذج وخوارزميات متطورة وقوية لفهرسة واسترجاع وتنظيم الصور وبيانات الوسائط المتعددة والتفاعل معها.
The explosion of image data on the Internet has the potential to foster more sophisticated and robust models and algorithms to index, retrieve, organize and interact with images and multimedia data.

إلا أن كيفية تسخير وتنظيم هذه البيانات لا تزال تُمثل مشكلةً حرجة.
But exactly how such data can be harnessed and organized remains a critical problem.

تُقدم هذه الورقة قاعدة بيانات جديدة تُسمى **شبكة الصور**، ك**هيكل مفاهيم** واسع النطاق من الصور المبني على اساس **شبكة الكلمات**.
We introduce here a new database called “ImageNet”, a largescale ontology of images built upon the backbone of the WordNet structure.

تهدف **شبكة الصور** إلى ملء غالبية **مجموعات المرادفات** التي تقدمها **شبكة الكلمات**، والبالغ عددها 80,000، بصور نقية وكاملة الدقة بمتوسط ​​يتراوح بين 500 و1000.
ImageNet aims to populate the majority of the 80,000 synsets of WordNet with an average of 500-1000 clean and full resolution images.

سيؤدي ذلك إلى عشرات الملايين من الصور المعنونة، والمُرتبة حسب التسلسل الهرمي الدلالي لـ **شبكة الكلمات**.
This will result in tens of millions of annotated images organized by the semantic hierarchy of WordNet.

تُقدم هذه الورقة تحليلًا مُفصلًا لـ **شبكة الصور** في حالتها الحالية: المؤلفة من 12 شجرة فرعية تحتوي على 5247 مجموعة مرادفات و3.2 مليون صورة إجمالًا.
This paper offers a detailed analysis of ImageNet in its current state: 12 subtrees with 5247 synsets and 3.2 million images in total.

وضح الباحثون أن **شبكة الصور** أكبر حجمًا وتنوعًا، وأكثر دقةً من مجموعات الصور الحالية.
We show that ImageNet is much larger in scale and diversity and much more accurate than the current image datasets.

وحيث أن إنشاء قاعدة بيانات ضخمة كهذه مهمةً صعبة.
Constructing such a large-scale database is a challenging task.

فسيشرح الباحثون آلية جمع البيانات باستخدام Amazon Mechanical Turk.
We describe the data collection scheme with Amazon Mechanical Turk.

وأخيرًا، سيوضحون فائدة **شبكة الصور** من خلال ثلاثة تطبيقات بسيطة في التعرف على الكائنات، وتصنيف الصور، وتجميع الكائنات تلقائيًا.
Lastly, we illustrate the usefulness of ImageNet through three simple applications in object recognition, image classification and automatic object clustering.

يأمل الباحثون أن يُتيح حجم **شبكة الصور** ودقتها وتنوعها وبنيتها الهرمية فرصًا لا مثيل لها للباحثين في مجال الرؤية الحاسوبية ومجالات أخرى.
We hope that the scale, accuracy, diversity and hierarchical structure of ImageNet can offer unparalleled opportunities to researchers in the computer vision community and beyond.


## 1\. مقدمة

جاء العصر الرقمي بطفرة هائلة في البيانات.
The digital era has brought with it an enormous explosion of data.

حيث تشير أحدث التقديرات إلى وجود أكثر من 3 مليارات صورة على فيكر، وعدد مماثل من مقاطع الفيديو على يوتيوب، وعدد أكبر من الصور على محرك بحث الصور من قوقل.
The latest estimations put a number of more than 3 billion photos on Flickr, a similar number of video clips on YouTube and an even larger number for images in the Google Image Search database.

يُمكن تطوير نماذج وخوارزميات أكثر تطورًا وقوة من سابقاتها باستغلال هذا العدد الهائل من الصور، وإنتاج تطبيقات أفضل لفهرسة هذه البيانات واسترجاعها وتنظيمها والتفاعل معها.
More sophisticated and robust models and algorithms can be proposed by exploiting these images, resulting in better applications for users to index, retrieve, organize and interact with these data.

تبقى كيفية الاستفادة من هذه البيانات وتنظيمها مشكلة لم تُحل بعد.
But exactly how such data can be utilized and organized is a problem yet to be solved.

يقدم الباحثون في هذه الورقة **شبكة الصور** كقاعدة بيانات صور جديدة، و**هيكل مفاهيم** واسع للصور.
In this paper, we introduce a new image database called “ImageNet”, a large-scale ontology of images.

ويعتقد الباحثون أن **هيكل المفاهيم** الواسع للصور يُعد موردًا بالغ الأهمية لتطوير خوارزميات متقدمة وواسعة النطاق للبحث عن الصور وفهمها بناء على محتواها، بالإضافة إلى توفير بيانات تدريب وقياس أداء تلك الخوارزميات.
We believe that a large-scale ontology of images is a critical resource for developing advanced, large-scale content-based image search and image understanding algorithms, as well as for providing critical training and benchmarking data for such algorithms.

--------------

تستخدم **شبكة الصور** البنية الهرمية **شبكة الكلمات**[^9].
ImageNet uses the hierarchical structure of WordNet [9].

حيث يُطلق على كل مفهوم ذي معنى في **شبكة الكلمات**، والذي ربما يتم وصفه بكلمات أو عبارات متعددة، اسم **مجموعة مرادفات**.
Each meaningful concept in WordNet, possibly described by multiple words or word phrases, is called a “synonym set” or “synset”.

تحتوي **شبكة الكلمات** على حوالي 80,000 **مجموعة مرادفات** اسمية.
There are around 80; 000 noun synsets in WordNet.

يهدف الباحثون في هذه الورقة لتوفير ما متوسطه 500-1000 صورة لكل **مجموعة مرادفات**.
In ImageNet, we aim to provide on average 500-1000 images to illustrate each synset.

تُراقب جودة صور كل مفهوم ويتم شرحها بواسطة إنسان كما هو موضح في القسم 3.2.
Images of each concept are quality-controlled and human-annotated as described in Sec. 3.2.

وستوفر **شبكة الصور** بذلك، عشرات الملايين من الصور المرتبة بشكل نظيف.
ImageNet, therefore, will offer tens of millions of cleanly sorted images.

يقدم الباحثون في هذه الورقة، الإصدار الحالي من **شبكة الصور**، والذي يتكون من 12 "شجرة فرعية": لـ الثدييات والطيور والأسماك والزواحف والبرمائيات والمركبات والأثاث والآلات الموسيقية والتكوينات الجيولوجية والأدوات والزهور والفواكه.
In this paper, we report the current version of ImageNet, consisting of 12 “subtrees”: mammal, bird, fish, reptile, amphibian, vehicle, furniture, musical instrument, geological formation, tool, flower, fruit.

تحتوي هذه **الأشجار الفرعية** على 5,247 **مجموعة مرادفات** و3.2 مليون صورة.
These subtrees contain 5247 synsets and 3:2 million images.

يوضح الشكل 1 لقطة لفرعين من الأشجار الفرعية الثدييات والمركبات.
Fig. 1 shows a snapshot of two branches of the mammal and vehicle subtrees.

أتاح الباحثون قاعدة البيانات للعامة على [التالي](http://www.image-net.org).
The database is publicly available at http://www.image-net.org.

----

رتب الباحثون ما تبقى من هذه الورقة على النحو التالي: أولاً: (القسم 2) يبين **شبكة الصور** كقاعدة بيانات صور واسعة النطاق ودقيقة ومتنوعة.
The rest of the paper is organized as follows: We first show that ImageNet is a large-scale, accurate and diverse image database (Section 2).

وفي القسم 4، يُقدِّم الباحثون بعض الأمثلة التطبيقية البسيطة من خلال استغلال شبكة الصور الحالية، وخاصةً الأشجار الفرعية للثدييات والمركبات.
In Section 4, we present a few simple application examples by exploiting the current ImageNet, mostly the mammal and vehicle subtrees.

بهدف إثبات أن شبكة الصور تُشكل موردًا مفيدًا لتطبيقات التعرّف البصري، مثل التعرّف على الكائنات وتصنيف الصور وتحديد مواقعها.
Our goal is to show that ImageNet can serve as a useful resource for visual recognition applications such as object recognition, image classification and object localization.

بالإضافة إلى أن بناء قاعدة بيانات واسعة النطاق وعالية الجودة، لم يعد ممكناً بالاعتماد على أساليب جمع البيانات التقليدية.
In addition, the construction of such a large-scale and high-quality database can no longer rely on traditional data collection methods.

يصف القسم 3 كيفية بناء شبكة الصور بالاستفادة من Amazon Mechanical Turk.
Sec. 3 describes how ImageNet is constructed by leveraging Amazon Mechanical Turk.


## 2\. خصائص شبكة الصور

بُنيت شبكة الصور على النية الهرمية التي تقدمها شبكة الكلمات.
ImageNet is built upon the hierarchical structure provided by WordNet.

وتهدف للصول إلى ما يقارب 50 مليون صورة، عند اكتمالها، كاملة الدقة ومُعنونة بوضوح، على أن تحتوي كل مجموعة مرادفات مابين 500 و 1000 صورة.
In its completion, ImageNet aims to contain in the order of 50 million cleanly labeled full resolution images (500-1000 per synset).

تتكون شبكة الصور حتى وقت كتابة هذه الورقة من 12 شجرة فرعية.
At the time this paper is written, ImageNet consists of 12 subtrees.

ستعتمد معظم التحليلات في هذا البحث على شجرتين فرعيتين هما الثديات والمركبات.
Most analysis will be based on the mammal and vehicle subtrees.

---

**النطاق**
Scale

تهدف شبكة الصور إلى توفير تغطية شاملة ومتنوعة لعالم الصور.
ImageNet aims to provide the most comprehensive and diverse coverage of the image world.

تحتوي الاشجار الفرعية الـ 12 الحالية على إجمالي 3.2 مليون صورة معنونة بدقة، وموزعة على 5,247 صنف (الشكل 2).
The current 12 subtrees consist of a total of 3:2 million cleanly annotated images spread over 5247 categories (Fig. 2).

جمعنا - في المتوسط - أكثر من 600 صورة لكل مجموعة مفردات
On average over 600 images are collected for each synset.

يوضح الشكل 2 توزيع عدد الصور لكل مجموعة مرادفات في شبكة الصور الحالية[^a].
Fig. 2 shows the distributions of the number of images per synset for the current ImageNet [^a].

تُعد هذه أكبر مجموعة بيانات صور مُوَصّفة متاحة لمجتمع أبحاث الرؤية الحاسوبية، حسب علم الباحثين، من حيث العدد الإجمالي للصور، وعدد الصور لكل فئة، بالإضافة إلى عدد الفئات.
To our knowledge this is already the largest clean image dataset available to the vision research community, in terms of the total number of images, number of images per category as well as the number of categories [^b].

-----
الشكل 1: لقطة لشجرتين فرعييتين من شبكة الصور من الجذر إلى الأوراق: الصف العلوي من الشجرة الفرعية للثدييات، والصف السفلي من الشجرة الفرعية للمركبات.
Figure 1: A snapshot of two root-to-leaf branches of ImageNet: the top row is from the mammal subtree; the bottom row is from the vehicle subtree.

تعرض 9 صور مختارة عشوائياً لكل مجموعة مرادفات.
For each synset, 9 randomly sampled images are presented.

-----
الشكل 2: نطاق شبكة الصور. يوضح المنحنى الأحمر في رسم بياني عدد الصور لكل مجموعة مرادفات.
Figure 2: Scale of ImageNet. Red curve: Histogram of number of images per synset.

حوالي 20% من مجموعات المرادفات تحتوي على عدد قليل جدًا من الصور.
About 20% of the synsets have very few images. 

أكثر من 50% من مجموعات المرادفات تحتوي على أكثر من 500 صورة.
Over 50% synsets have more than 500 images.

يلخص الجدول الأشجار الفرعية مختارة.
Table: Summary of selected subtrees

للاطلاع على الإحصائيات الكاملة والمحدثة، تفضل بزيارة [الموقع](http://www.image-net.org/about-stats).
For complete and up-to-date statistics visit http://www.image-net.org/about-stats.

------
الشكل 3: مقارنة بين شجرتي "القطط" و"الماشية" الفرعيتين في لعبة ESP [25] وشبكة الصور.
Figure 3: Comparison of the “cat” and “cattle” subtrees between ESP [25] and ImageNet.

يتناسب حجم العقدة في كل شجرة مع عدد الصور التي تحتويها.
Within each tree, the size of a node is proportional to the number of images it contains.

ويُظهر الشكل عدد الصور لأكبر عقدة في كل شجرة.
The number of images for the largest node is shown for each tree.

أما العقد المشتركة بين شجرة ESP وشجرة شبكة الصور فهي مُلوّنة باللون الأحمر.
Shared nodes between an ESP tree and an ImageNet tree are colored in red.

## الحواشي
[^a]: حوالي 20% من مجموعات المرادفات تحتوي على عدد قليل جداً من الصور، إما بسبب قلة الصور المتاحة على الإنترنت، مثل "خفاش مسائي"، أو لصعوبة توضيح مجموعة المرادفات بالصور، مثل "حصان عمره سنتين".
[^b]: تزعم لعبة ESP أنها قد صنفت عدداً كبيراً جداً من الصور، ولكن جزء صغير حوالي 60 ألف صورة فقط متاحة للعامة.


## المراجع
[^9]: C. Fellbaum. WordNet: An Electronic Lexical Database. Bradford Books, 1998.
[^25]: L. von Ahn and L. Dabbish. Labeling images with a computer game. In CHI04, pages 319–326, 2004.

[المصدر](https://web.archive.org/web/20210115185228if_/http://www.image-net.org/papers/imagenet_cvpr09.pdf)

