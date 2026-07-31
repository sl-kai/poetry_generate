"""
诗歌生成器模块
基于 Transformer + 平水韵押韵约束 + Beam Search
"""

import math
import random

from .transformer_model import TransformerModel
from .tone_dict import ToneDict, WUYAN_TEMPLATES, QIYAN_TEMPLATES


# ============================================================
# 平水韵表
# ============================================================
POEMYUN = {
    'a': '啊腌扒叭巴吧芭岜疤笆粑豝嚓叉杈差咖瓜抓胍榻喇哈阿花哗加茄迦痂枷耞珈袈嘉佳家傢葭豭咖夸姱啦妈摩嬷趴凹咱葩杉沙莎痧鲨纱砂他她它哇洼蛙娲虾丫呀鸦哑桠查楂喳呱欻旮笳拉垃吗蚂仨裟砂趿渣揸馇挝阿八捌擦插锸耷哒嗒鎝褡发筏夹嘎刮栝鸹拉邋抹掐袷葜撒杀刹铩煞刷趿塌溻禢踏挖呷瞎鸭压押扎匝咂拶吒哳浃撒答啊茶查搽嵖猹槎楂碴苴垞蛤华哗骅铧麻嘛蟆拿扒杷爬钯耙筢琶娃霞遐瑕暇牙伢芽岈玡蚜崖涯睚衙拔茇菝跋魃察茬檫达鞑沓怛妲笪靼答跶乏伐茷垡砝阀罚嘎滑划猾夹浃郏荚铗蛱恝戛颊旯拉匣狎挟柙侠峡狭硖辖黠杂砸扎札轧闸铡喋剳',
    'o': '波播菠玻嶓搓磋蹉瑳多哆呙锅卜埚涡坡颇摸阿陂莎唆娑梭挲疙睃圪嗦嗍鲅蓑拖莴倭唷涡窝蜗踒阿婀痾哥歌戈呵科蝌柯疴苛珂窠轲颗屙菏棵髁的了么呢车奢赊畲遮仡猞拨鱿趵钵饽剥逴踔戳撮咄剟掇裰郭崞聒蝈豁劐擢捋泊泼钹说缩托脱喔拙捉苛桌倬涿焯作嘬鸽割搁喝磕瞌榼脖嵯痤瘥鹾罗萝啰逻脶猡锣椤箩骡螺谟馍馍摹模麽摩磨嬷蘑磨魔挪娜傩婆鄱繁皤驮佗陀坨驼柁砣鸵酡跎蛇鼍鹅蛾娥莪俄峨哦讹和禾何河荷阇膜婆皤沱渮哪挼孛荸伯驳帛瓝泊柏勃钹铂舶博鹁浡渤搏箔膊踣薄馞欂襮礴夺度铎踱怫佛掇咄裰剟国掴帼漍腘虢馘活嗑橐灼彴茁踔卓斫浊酌浞诼着啄琢椓蠋擢濯镯昨作笮阁葛蛤颌合涸盒膜拙棁捽貉曷盍阖壳德得额鹖则舴晢蜇革格鬲隔嗝槅膈滆塥鎘骼纥劾阂核翮壳咳颏舌则责择咋泽啧帻舴箦赜折哲辄蛰谪摺磔辙翟',
    'ie': '爹阶皆喈嗟街湝乜咩些靴耶倻椰偕掖瘪憋鳖跌颏疙节截疖角圪结接秸揭噘撅捏撇瞥切缺阙贴得怗贴帖楔歇蝎削薛噎曰约瘸斜邪偕谐鞋携爷耶茄伽鲑踅椰别蹩迭垤昳絰瓞谍堞耋揲喋牒叠碟蝶艓蹀孑节讦劫劼杰诘拮洁结桔桀捷偈玦觉决绝倔桷掘崛脚觖厥劂谲獗蕨橛噱爵蹶矍嚼爝攫钁协胁挟絜颉撷勰襭穴学噱',
    'ai': '开哎哀埃挨娭唉欸掰偲钗差揣呆该陔垓荄赅乖揩腮毸鳃筛酾衰摔苔台胎歪灾哉栽甾斋拍摘拆塞猜挨皑癌才财材裁侪柴豺还孩骸徊怀淮槐踝来莱崃徕涞埋霾俳排徘牌簰邰苔抬骀炱鲐白宅翟',
    'ei': '微欸陂杯卑背悲碑鹎衰崔催摧縗吹榱炊堆飞妃非菲啡騑绯馡蜚扉霏鲱归圭龟妫规邽扳闺硅黑嘿傀瘣瑰鲑灰诙虺挥咴恢袆珲豗晖辉翬麾徽亏刲岿勒悝盔窥胚呸绥衃醅虽荽睢濉忒推危威逶偎隈葳椳煨溦巍蝛薇追骓锥椎欸垂陲捶椎槌棰倕锤箽肥淝腓回茴洄蛔奎逵馗隗葵揆骙暌魁戣睽蝰累雷嫘缧擂檑礌镭羸罍虆玫枚眉莓脢梅嵋猸湄邳媒楣煤酶镅霉糜陪培赔没裵蕤绥隋随遂谁颓韦为圩贼违围帏沩桅唯帷惟薇维嵬巍潍闱',
    'ao': '凹熬包薄苞胞孢剥龅煲褒标飚彪骠镖瘭飙操糙藨镳瀌膘杓猋骉摽操抄怊钞超剿潦刀叨忉刁汈蛁雕貂叼碉凋鲷高皋羔槔睾膏篙糕蒿薅嚆交姣骄艽郊茭浇娇胶椒蛟焦蕉教跤僬鲛嶕礁噍鹪轇尻捞撩猫喵孬抛脬泡漂剽慓飘缥螵僄悄磽磽锹劁敲橇缲搔骚挑缫臊捎烧梢稍筲艄蛸叨涛绦掏滔韬弢饕慆佻祧肖枭枵哓骁嘐逍虓鸮消宵绡萧硝销削蠨蛸翛箫潇霄魈歊嚣幺约夭吆妖要喓腰邀遭糟钊招昭嘲啁着朝约剥削豪敖璈遨嗷廒獒熬隞嶅聱翱鳌鏖螯骜薄雹曹槽螬漕嘈晃巢朝嘲潮捡捯号嗥毫壕濠嚎蠔嚼劳崂痨牢捞唠醪聊辽疗僚漻寥嘹獠寮缭嫽燎憭髎毛矛茅牦旄酕锚髦蝥蟊苗描瞄饶挠饶蛲猱呶刨咆狍庖炮袍匏嫖朴瓢薸乔侨荞峤桥硚翘谯荍鞽憔樵瞧荛桡蛲饶娆桡蛲饶娆苕韶勺芍咷梼逃洮桃陶萄梼啕淘綯醄鼗条岹苕调笤齠蜩迢髫岧鲦崤淆洨爻尧肴淆轺峣陶姚窑谣摇徭遥猺瑶飖鳐轺凿着',
    'ou': '抽掐紬瘳丢都兜蔸勾佝沟枸钩缑篝巘鞲齁勼纠鸠究赳阄湫揪啾蝤轇芤抠眍溜熘搂喽哞妞区讴沤瓯欧殴呕鸥丘邱龟秋蚯湫楸鹙鳅鞦收搜嗖锼馊廋溲飕艘偷修脩休咻羞鸺貅馐髹优攸忧悠呦幽麀舟州谄侜周洲粥啁赒輈诪邹郰緅驺诹陬鲰粥儔帱畴筹酬俦踌惆绸稠禂仇愁雠侯喉猴篌瘊糇流留榴骝刘浏瘤琉硫旒鹠遛镏飗瘤鎏娄楼偻蒌喽耧蝼髅牟眸谋蛑缪鍪牛抔掊裒囚仇犰求毬虬泅俅璆酋逑球遒赇裘璆蝤柔揉糅煣蹂鞣头投骰尤犹疣鱿莸铀由邮油游猷繇蝣妯轴',
    'an': '安氨唵桉庵谙鹌鞍盦扳班颁斑攽般搬瘢皤癍参骖餐觇搀幨襜川穿氽撺镩丹担单眈酖耽郸聃禅儋殚箪端帆番蕃幡藩翻干甘杆玕肝柑竿疳尴关观纶官冠矜倌棺瘝鳏顸酣憨鼾欢讙獾驩刊看勘龛堪戡宽髋颟囡番潘攀三叁山芟杉删衫姗珊栅舢扇跚煽潸膻闩拴栓酸坍贪摊滩瘫湍弯剜湾蜿豌糌簪占沾毡旃粘詹谵邅瞻专砖颛钻躜边砭萹笾编煸蝙鯿箯鞭参骖餐掂傎癫滇颠巅戋尖奸歼坚间肩艰监兼菅笺渐犍湔缄蒹煎缣鹣搛熸鞬鳒韉捐涓娟朘圈鹃镌蠲拈蔫片扁偏篇犏翩千仟阡芊扦迁佥钎牵铅悭谦签愆鹐骞搴磏諐褰圈悛棬弮天添黇仙先纤氙忺籼掀铦酰跹锨鲜暹骞轩宣谖萱揎喧瑄煖禤暄儇懁咽恹殷胭烟焉崦阉阏奄淹腌湮鄢嫣燕鸢眢鸳冤渊鹓箢寒残蚕惭单鋋馋谗婵禅孱缠蝉廛僝潺澶镡蟾镵巉躔传船遄椽攒凡矾烦墦蕃攀樊璠燔繁蘩邗汗邯含函琀焓晗涵韩还环桓圜阛寰缳鬟郇荁洹貆澴轘兰岚斓拦栏婪阑蓝谰澜褴篮斕镧峦娈孪挛鸾脔滦栾銮蛮谩蔓馒瞒鞔鳗鬘男南难喃楠爿胖般盘磻磐蹒蟠蚺然燃髯坛昙倓郯谈弹覃谭痰潭檀团漙抟咱丸纨完玩顽刓汍烷奁连镰怜帘莲涟联裢鲢廉濂鐮鬑磏眠绵棉年粘黏鲇便骈胼蹁钤前虔钱钳乾掮潜黔犍权全佺诠荃泉辁拳铨痊惓筌蜷醛鳈鬈颧田佃畋恬钿甜湉填阗闲贤弦咸挦涎娴衔舷痫鹇嫌玄悬旋漩璇延蜒严言芫妍岩炎沿铅研盐阎筵颜檐元园员沅垣湲袁原圆鼋援媛缘猿塬嫄源羱辕橼',
    'en': '奔贲锛玢宾彬傧斌滨缤槟濒豳参抻郴伧琛嗔汶瞋春椿蝽村皴踆惇吨墩礅敦蹲恩分芬吩纷玢菌氛棻雰根跟昏阍惛婚巾斤今紷金津衿矜筋禁襟军均龟君钧皲坤昆崑裈堃焜琨髡鹍锟鲲抡拎闷喷拼姘钦侵亲衾駸嵚囷逡森申伸身呻砷侁诜参绅珅莘娠深糁燊孙荪狲飧吞暾温瘟心芯骎辛忻昕欣炘锌新歆薪馨鑫勋埙熏薰薫獯曛醺窨因阴茵洇裀荫音姻氤殷堙喑闉愔禋晕缊氲煴赟贞针侦浈珍帧胗真桢砧祯蓁斟甄獉溱榛箴臻迍肫窀谆尊遵樽鳟岑涔臣尘辰沉忱陈宸晨谌纯莼唇淳鹑漘醇存蹲坟汾棼焚濆痕贲贲浑珲馄混哏魂邻林临淋琳粼磷潾嶙遴霖辚瞵鳞麟仑伦论抡沦纶轮门扪们民忞旻岷缗您盆湓贫频嫔颦芹芩矜秦琴覃禽勤懃擒噙螓裙群人壬仁任神什屯囤饨豚臀文纹炆闻蚊雯旬郇寻巡询洵荀荨峋恂鲟循吟垠龈狺訚崟银荧淫寅蟫鄞夤嚚霪云匀芸员沄纭昀畇筠耘筼麇',
    'ang': '肮邦帮梆浜仓苍沧鸧舱昌倡菖猖阊娼伥创疮窗胯当珰铛裆筜方坊芳枋邡钫冈岗刚矼肛纲钢缸釭罡堽光咣胱夯荒肓塃慌江将姜豇浆僵螀缰疆康慷糠囊匡劻诓恇筐牤乓雱滂膀枪锵羌戗戕将腔蜣铰镪嚷丧桑伤汤殇商觞墒熵双泷霜孀孀骦鹴汤铴耥嘡镗蹚汪乡芗相香厢湘缃箱襄骧镶秧央湍殃鸯鞅赃脏臧张章獐彰嫜璋樟蟑妆庄桩装卬昂藏长场苌肠尝常偿徜裳嫦床幢防妨坊忍肪鲂房行吭迒杭絎航颃皇黄凰隍喤煌遑徨湟惶粕锽潢璜蝗篁磺蟥簧鳇扛狂诳鵟郞狼郎琅榔桹琅廊嫏樃硠锒稂鎯螂良俍莨凉梁椋量粮粱邙芒忙杧盲氓茫硭铓牻嚢馕娘彷庞逄旁蒡膀磅螃强墙蔷嫱樯蘘灢禳瓤唐堂棠塘搪糖溏瑭樘膛赯螗螳鄌亡王详降绛庠祥翔扬阳羊玚飏炀杨旸佯疡徉洋',
    'eng': '庚更横盛正应乘胜兴行刑莹茔宁凭暝钉称令并伻崩听祊绷嘣冰兵槟屏柽琤称蛏铛赪撑噌瞠灯登噔蹬镫丁仃叮玎盯钉疔酊靪丰风封枫疯峰烽葑锋蜂更庚耕赓鹒羹亨哼精茎惊京经睛泾荆菁旌晶粳兢鲸坑吭硁铿蒙抨怦砰烹嘭乒俜娉青轻氢倾卿圊清蜻鲭扔僧升生声牲笙甥鼪厅汀听翁嗡兴星狚惺腥应英莺婴撄嘤罂缨璎樱鹦媖瑛膺鹰曾增憎缯罾正争筝蒸征怔挣峥狰钲症烝睁铮稳东冲憧充忡翀舂惮艟匆苁囱枞葱骢璁聪熜冬咚鸫工弓公功攻供肱宫恭蚣躬龚觥哼轰哄訇烘薨埛駉扃空倥崆屄箜忪松凇菘嵩恫通嗵瘑凶兄芎匈汹恟胸佣痈拥邕鄘雍墉慵庸镛壅臃鳙中忪忠终钟盅衷螽宗综棕踪鬃层曾嶒成酲丞呈枨诚承城宬乘盛程惩裎塍澄橙冯逢缝恒姮桁珩横衡蘅楞棱伶灵苓蛉囹泠玲令瓴铃鸰凌陵聆菱棂暹舲翎羚绫棱零龄鲮酃氓虻萌蒙盟甍瞢幪濛曚矇朦艨檬名茗明鸣冥铭洺蓂溟暝瞑螟能棕拧咛狞柠凝芃仍朋膨堋澎彭棚蓬硼鹏篷鬅平冯评坪苹凭枰洴帡屏瓶萍勍情晴檠擎黥绳渑疼腾誊螣藤廷亭停庭蜓婷霆行形邢陉型荥盈萤莹营萦楹滢蝇潆贏赢瀛迎虫重崇从丛尝悰琮弘红吰闳宏泓荭虹闳洪翃鸿黉龙栊茏咙泷珑眬胧昽聋笼隆癃窿农侬哝傢浓脓秾邛穷茕穹藭筇琼蛩跫戎茸荣绒容崂蓉溶瑢榕融嵘同彤侗苘峒桐砼垌佟烔鲷峂樟僮铜童潼瞳朣曈艟雄熊喁颙',
}
# 平水韵字典：char → rhyme_group
PINGSHUI = {}
for group, chars in POEMYUN.items():
    for c in chars:
        PINGSHUI[c] = group

# 诗歌结构
POEM_STRUCT = {
    '五言绝句': {'rhyme_pos': [10, 22], 'line_len': 5, 'lines': 4},
    '五言律诗': {'rhyme_pos': [10, 22, 34, 46], 'line_len': 5, 'lines': 8},
    '七言绝句': {'rhyme_pos': [14, 30], 'line_len': 7, 'lines': 4},
    '七言律诗': {'rhyme_pos': [14, 30, 46, 62], 'line_len': 7, 'lines': 8},
}


class PoemGenerator:
    """古诗生成器 —— 平水韵押韵 + Beam Search"""

    def __init__(self, transformer_model=None, tone_dict=None, analyzer=None):
        self.transformer = transformer_model or TransformerModel()
        self.tone = tone_dict or ToneDict()
        self.analyzer = analyzer or PoetryAnalyzer()

    def _predict_next(self, prefix, n=20, temperature=0.8):
        return self.transformer.predict_next(prefix, n=n, temperature=temperature)

    def _score_sequence(self, text):
        return self.transformer.score_sequence(text)

    # ============================================================
    # 共享方法
    # ============================================================

    def _compute_scores(self, lines, poem_type):
        """计算流畅度和平仄得分（公共方法，避免重复）"""
        log_probs = [self._score_sequence(l) for l in lines]
        fluency = sum(max(lp, -6) for lp in log_probs) / len(log_probs)
        fluency_score = max(0, min(100, (fluency + 6) / 4 * 100))
        tone_scores = []
        for li, l in enumerate(lines):
            t = self._get_tone_template(poem_type, li)
            if t:
                _, _, r = self.tone.check_compliance(l, t)
                tone_scores.append(r)
        avg_tone = sum(tone_scores) / len(tone_scores) if tone_scores else 0
        return fluency_score, avg_tone

    # ============================================================
    # 关键词生成
    # ============================================================

    def generate_from_keywords(self, keywords, poem_type='五言绝句',
                                beam_width=10, temperature=0.8):
        struct = POEM_STRUCT[poem_type]
        line_len, total_lines = struct['line_len'], struct['lines']

        print(f'\n{"="*50}')
        print(f'关键词生成: {" ".join(keywords)} → {poem_type}')
        print(f'{"="*50}')

        lines, seen_chars = [], set()
        kw_chars = set(''.join(keywords))
        kw_multi = [kw for kw in keywords if len(kw) >= 2]
        kw_placed = set()

        # 预选韵部：找含常用字多的韵部
        common_ends = [c for c, _ in self.analyzer.line_end_freq.most_common(500)]
        rhyme_scores = {}
        for group, chars in POEMYUN.items():
            cnt = sum(1 for c in common_ends if c in chars)
            if cnt >= 5:
                rhyme_scores[group] = cnt
        rhyme_group = max(rhyme_scores, key=rhyme_scores.get) if rhyme_scores else None
        if rhyme_group:
            print(f'  预选韵部: {rhyme_group}')

        for li in range(total_lines):
            is_rhyme = li in [1, 3, 5, 7][:total_lines//2]
            template = self._get_tone_template(poem_type, li)

            # 每行只分配一个未放置的关键词，且该词未在任何已生成行中出现
            force_kw = next((kw for kw in kw_multi
                            if kw not in kw_placed
                            and not any(kw in l for l in lines)), None)

            beam = self._beam_search_line(
                line_len=line_len, beam_width=beam_width, temperature=temperature,
                tone_template=template, seen_chars=seen_chars,
                previous_lines=lines, is_rhyme=is_rhyme, rhyme_group=rhyme_group,
                force_kw=force_kw, kw_chars=kw_chars,
            )

            # 如果关键词没命中任何行，强制用注入行
            if force_kw and force_kw not in kw_placed:
                injected = self._inject_keyword(force_kw, line_len, beam_width, temperature,
                                                 template, seen_chars, lines,
                                                 is_rhyme, rhyme_group)
                if injected:
                    line = injected[0]
                    lines.append(line)
                    for c in line:
                        seen_chars.add(c)
                    if force_kw in line:
                        kw_placed.add(force_kw)
                    if is_rhyme and rhyme_group is None:
                        rhyme_group = PINGSHUI.get(line[-1])
                        if rhyme_group:
                            print(f'  押韵: {rhyme_group}部')
                    print(f'  第{li+1}句[注入]: {line}')
                    continue

            line = beam[0][0] if beam else self._fallback(line_len)
            if force_kw and force_kw in line:
                kw_placed.add(force_kw)
            lines.append(line)
            for c in line:
                seen_chars.add(c)
                kw_chars.discard(c)  # 出现过的关键词字不再奖励

            # 记录韵部
            if is_rhyme and rhyme_group is None:
                rhyme_group = PINGSHUI.get(line[-1])
                if rhyme_group:
                    print(f'  押韵: {rhyme_group}部 ({POEMYUN[rhyme_group][:20]}...)')

            print(f'  第{li+1}句: {line}')

        poem = '\n'.join(lines)
        fluency_score, avg_tone = self._compute_scores(lines, poem_type)
        print(f'\n{poem}')
        return {'poem': poem, 'lines': lines, 'score': fluency_score, 'tone_score': avg_tone}

    # ============================================================
    # 藏头诗生成
    # ============================================================

    def generate_acrostic(self, head_chars, poem_type=None,
                           beam_width=10, temperature=0.8):
        head = [c for c in head_chars if '一' <= c <= '鿿']
        n = len(head)

        if poem_type is None:
            poem_type = '五言绝句' if n <= 4 else '七言绝句'
        struct = POEM_STRUCT[poem_type]
        line_len, total_lines = struct['line_len'], struct['lines']
        if n > total_lines:
            head = head[:total_lines]

        print(f'\n{"="*50}')
        print(f'藏头诗: {"".join(head)} → {poem_type}')
        print(f'{"="*50}')

        lines, seen_chars = [], set()
        # 预选韵部
        common_ends = [c for c, _ in self.analyzer.line_end_freq.most_common(500)]
        rhyme_scores = {}
        for group, chars in POEMYUN.items():
            cnt = sum(1 for c in common_ends if c in chars)
            if cnt >= 5:
                rhyme_scores[group] = cnt
        rhyme_group = max(rhyme_scores, key=rhyme_scores.get) if rhyme_scores else None

        for li in range(total_lines):
            is_rhyme = li in [1, 3, 5, 7][:total_lines//2]
            template = self._get_tone_template(poem_type, li)
            prefix = head[li] if li < n else None

            beam = self._beam_search_line(
                line_len=line_len,
                beam_width=beam_width,
                temperature=temperature,
                tone_template=template,
                seen_chars=seen_chars,
                previous_lines=lines,
                is_rhyme=is_rhyme,
                rhyme_group=rhyme_group,
                prefix=prefix,
            )

            line = beam[0][0] if beam else self._fallback(line_len, prefix)
            lines.append(line)
            for c in line:
                seen_chars.add(c)

            if is_rhyme and rhyme_group is None:
                rhyme_group = PINGSHUI.get(line[-1])

            print(f'  第{li+1}句(藏头"{prefix}"): {line}')

        poem = '\n'.join(lines)
        fluency_score, avg_tone = self._compute_scores(lines, poem_type)
        print(f'\n{poem}')
        return {'poem': poem, 'lines': lines, 'head_chars': ''.join(head),
                'score': fluency_score, 'tone_score': avg_tone}

    # ============================================================
    # Beam Search（平水韵押韵约束）
    # ============================================================

    # 黑名单短语
    BANNED = {'不知何处', '何处是', '何处不', '何处有', '一年一', '一日一', '一度一',
              '不知何事', '知何处', '事不知', '何处寻', '何处'}

    def _beam_search_line(self, line_len, beam_width, temperature,
                           tone_template, seen_chars, previous_lines,
                           is_rhyme, rhyme_group, force_kw=None,
                           kw_chars=None, prefix=None):
        """Beam Search 生成一行"""
        prev_texts = set(previous_lines)
        kw_chars = kw_chars or set()

        if prefix:
            beam = [(prefix, 0.0)]
            start = len(prefix)
        else:
            starts = []
            for c, f in self.analyzer.line_start_freq.most_common(beam_width * 5):
                bonus = 3.0 if c in kw_chars else 0
                starts.append((c, math.log(max(f, 1)) + bonus))
            starts.sort(key=lambda x: -x[1])
            beam = starts[:beam_width * 2]
            start = 1

        for pos in range(start, line_len):
            candidates = []
            for text, score in beam:
                preds = self._predict_next(text, n=beam_width, temperature=temperature)
                for ch, prob in preds:
                    new_text = text + ch
                    new_score = score + math.log(max(prob, 1e-10))

                    # 黑名单检查
                    banned = False
                    for bp in self.BANNED:
                        if bp in new_text:
                            banned = True
                            break
                    if banned:
                        continue

                    # 字重复惩罚
                    if ch in new_text[:-1]:
                        new_score -= 10.0
                    if ch in seen_chars:
                        new_score -= 3.0

                    # 关键词奖励：只奖尚未出现的字
                    if ch in kw_chars and ch not in seen_chars:
                        new_score += 2.0
                    if force_kw and force_kw in new_text:
                        new_score += 5.0

                    # 平仄
                    if tone_template:
                        ts = self.tone.score_tone_match(ch, pos, tone_template)
                        new_score += 0.3 * math.log(max(ts, 0.1))

                    candidates.append((new_text, new_score))

            candidates.sort(key=lambda x: -x[1])
            beam = candidates[:beam_width]

        # 最终评分：押韵硬约束 + 黑名单
        final = []
        for text, score in beam:
            if len(text) != line_len or text in prev_texts:
                continue
            # 黑名单再次检查
            if any(bp in text for bp in self.BANNED):
                continue

            # 押韵约束：同韵部奖，不同罚，非韵表轻微罚
            if is_rhyme and rhyme_group:
                rg = PINGSHUI.get(text[-1])
                if rg == rhyme_group:
                    score += 3
                elif rg is None:
                    score -= 2  # 不在平水韵表，轻微罚
                else:
                    score -= 8  # 不同韵部，中等罚

            final.append((text, score))

        final.sort(key=lambda x: -x[1])
        return final[:beam_width]

    def _get_tone_template(self, poem_type, line_index):
        if '五言' in poem_type:
            return WUYAN_TEMPLATES['仄起首句不入韵'][line_index % 4]
        return QIYAN_TEMPLATES['仄起首句不入韵'][line_index % 4]

    def _inject_keyword(self, kw, line_len, beam_width, temperature,
                          tone_template, seen_chars, previous_lines,
                          is_rhyme, rhyme_group):
        """强制注入关键词：尝试所有插入位置，填满其余字符"""
        prev_texts = set(previous_lines)
        best = None
        best_score = float('-inf')

        for pos in range(line_len - len(kw) + 1):
            # 生成前缀
            if pos > 0:
                pre_cands = self.transformer.generate_line(
                    start_char=None, line_length=pos, beam_width=3, temperature=temperature
                )
                prefixes = [p for p, _ in pre_cands if len(p) == pos]
            else:
                prefixes = ['']

            # 生成后缀
            suffix_len = line_len - pos - len(kw)
            if suffix_len > 0:
                suf_cands = self.transformer.generate_line(
                    start_char=kw[-1], line_length=suffix_len + 1, beam_width=3, temperature=temperature
                )
                suffixes = [s[1:] for s, _ in suf_cands if len(s) == suffix_len + 1 and s[0] == kw[-1]]
            else:
                suffixes = ['']

            for pre in prefixes[:2]:
                for suf in suffixes[:2]:
                    line = pre + kw + suf
                    if len(line) != line_len or line in prev_texts:
                        continue
                    if any(bp in line for bp in self.BANNED):
                        continue
                    if any(line.count(c) > 1 for c in line):
                        continue
                    score = 5.0
                    flu = self._score_sequence(line)
                    if flu > float('-inf'):
                        score += flu + 5
                    if tone_template:
                        _, _, r = self.tone.check_compliance(line, tone_template)
                        score += r * 3
                    if is_rhyme and rhyme_group:
                        if PINGSHUI.get(line[-1]) == rhyme_group:
                            score += 3
                    if score > best_score:
                        best_score = score
                        best = line

        return (best, best_score) if best else None

    def _fallback(self, line_len, head=None):
        chars = list('春风明月山水天地人花落知多少')
        if head:
            return head + ''.join(random.choice(chars) for _ in range(line_len - 1))
        return ''.join(random.choice(chars) for _ in range(line_len))


def format_poem_output(result, mode='keyword'):
    """格式化打印生成结果"""
    print()
    if mode == 'keyword':
        print(f'关键词: {" ".join(result.get("keywords", []))}')
    elif mode == 'acrostic':
        head = result.get('head_chars', '')
        print(f'藏头: {head}')
        # 高亮藏头字
        lines = result.get('lines', [])
        poem = result.get('poem', '')
        print()
        for i, line in enumerate(lines):
            if i < len(head):
                print(f'  {line[0]}' + line[1:])
            else:
                print(f'  {line}')
    else:
        lines = result.get('lines', [])
        for line in lines:
            print(f'  {line}')

    if mode != 'acrostic':
        lines = result.get('lines', [])
        print()
        for line in lines:
            print(f'  {line}')

    score = result.get('score', 0)
    tone = result.get('tone_score', 0)
    print(f'\n流畅度: {score:.0f} | 平仄符合率: {tone*100:.0f}%')
    print()
