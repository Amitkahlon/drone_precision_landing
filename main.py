import mujoco
import mujoco.viewer
import time

# 1. הגדרת הסביבה הפיזיקלית בפורמט XML
# המודל כולל תאורה, רצפה ירוקה-אפרפרה, וכדור אדום בגובה 2 מטרים
xml_model = """
<mujoco>
    <worldbody>
        <light pos="0 0 3" dir="0 0 -1"/>

        <geom name="floor" type="plane" size="10 10 0.1" rgba="0.8 0.9 0.8 1"/>

        <body name="ball" pos="0 0 2">
            <joint name="free_joint" type="free"/>
            <geom type="sphere" size="0.15" rgba="0.9 0.1 0.1 1" mass="1"/>
        </body>
    </worldbody>
</mujoco>
"""

# 2. טעינת המודל הפיזיקלי לתוך מנוע MuJoCo
model = mujoco.MjModel.from_xml_string(xml_model)
data = mujoco.MjData(model)

# 3. הרצת חלון התצוגה הגרפי בצורה פאסיבית (לא חוסמת)
print("מפעיל את סימולציית Hello World ב-MuJoCo...")
with mujoco.viewer.launch_passive(model, data) as viewer:
    # לולאת הריצה בזמן אמת - תמשיך לרוץ כל עוד החלון פתוח
    while viewer.is_running():
        # קידום הסימולציה הפיזיקלית בצעד זמן אחד קטן (mj_step)
        mujoco.mj_step(model, data)

        # סנכרון המצב הפיזיקלי החדש עם מה שרואים בחלון הגרפי
        viewer.sync()

        # השהייה קלה כדי שקצב הריצה יתאים לעין אנושית (בערך 100 פריימים בשנייה)
        time.sleep(0.01)

print("החלון נסגר, הסימולציה הסתיימה בהצלחה!")