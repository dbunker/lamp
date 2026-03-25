1 ~  :- d2(V1).
1 ~  :- d0(V1).
1 ~  :- d6(V1).
1 ~  :- d1(V1).
2 ~ d2(V1) :- d1(V1).
2 ~  :- d6(V1); not d2(V1).
2 ~  :- d0(V1); not d2(V1).
2 ~  :- d1(V1); not d2(V1).
2 ~  :- d1(V1); d6(V1).
2 ~  :- d0(V2); d2(V1).
2 ~  :- d0(V2); d6(V1).
2 ~ d2(V1) :- d6(V1).
2 ~  :- d0(V1); d1(V1).
2 ~  :- d0(V2); d1(V1).
2 ~  :- d0(V1); d6(V1).
2 ~  :- d2(V1); not d0(V1).
2 ~  :- d1(V2); d6(V1).
2 ~  :- d1(V1); not d0(V1).
2 ~  :- d6(V1); not d0(V1).
2 ~  :- d6(V1); not d1(V1).
2 ~  :- d2(V1); not d1(V1).
2 ~  :- d1(V1); d2(V1).
2 ~  :- d0(V1); not d1(V1).
2 ~ d2(V1) :- d0(V1).
2 ~  :- d0(V1); d2(V1).
2 ~  :- d1(V1); not d6(V1).
2 ~  :- d2(V1); d6(V1).
2 ~  :- d2(V1); not d6(V1).
2 ~  :- d1(V2); d2(V1).
2 ~  :- d0(V1); not d6(V1).
2 ~  :- d2(V2); d6(V1).
3 ~  :- d0(V3); d1(V2); d2(V1).
3 ~ d2(V1) :- d1(V1); d6(V2).
3 ~  :- d0(V2); d1(V1); not d2(V1).
3 ~ d2(V1) :- d0(V1); d6(V2).
3 ~  :- d1(V2); d6(V1); not d2(V1).
3 ~  :- d6(V1); not d0(V1); not d2(V1).
3 ~  :- d0(V1); d6(V1); not d2(V1).
3 ~ d2(V1) :- d0(V1); d1(V1).
3 ~ d2(V1) :- d0(V1); d2(V2).
3 ~  :- d6(V1); not d1(V1); not d2(V1).
3 ~ d2(V1) :- d0(V1); d1(V2).
3 ~ d2(V1) :- d1(V1); d2(V2).
3 ~  :- d0(V1); not d1(V1); not d2(V1).
3 ~ d2(V1) :- d2(V2); d6(V1).
3 ~  :- d0(V1); not d2(V1); not d6(V1).
3 ~  :- d1(V1); d6(V1); not d2(V1).
3 ~  :- d1(V1); not d0(V1); not d2(V1).
3 ~  :- d1(V1); d2(V2); d6(V1).
3 ~  :- d1(V2); d2(V1); d6(V1).
3 ~  :- d0(V1); d2(V2); d6(V1).
3 ~  :- d1(V3); d2(V2); d6(V1).
3 ~  :- d1(V2); d2(V2); d6(V1).
3 ~  :- d0(V3); d2(V2); d6(V1).
3 ~  :- d0(V1); d1(V1); not d2(V1).
3 ~ d2(V1) :- d1(V1); d6(V1).
3 ~  :- d1(V1); not d2(V1); not d6(V1).
3 ~  :- d0(V2); d6(V1); not d2(V2).
3 ~  :- d0(V2); d1(V1); not d2(V2).
3 ~  :- d1(V2); d6(V1); not d2(V2).
3 ~  :- d0(V2); d2(V1); not d1(V2).
3 ~  :- d0(V1); d2(V1); not d6(V1).
3 ~  :- d0(V1); d1(V2); d2(V1).
3 ~ d2(V1) :- d0(V1); not d6(V1).
3 ~  :- d0(V1); d1(V1); not d6(V1).
3 ~  :- d0(V2); d6(V1); not d1(V2).
3 ~  :- d0(V1); d2(V1); d6(V1).
3 ~  :- d1(V1); d2(V1); d6(V1).
3 ~ d2(V1) :- d1(V1); not d6(V1).
3 ~  :- d1(V1); d2(V1); not d6(V1).
3 ~  :- d0(V2); d2(V2); d6(V1).
3 ~  :- d0(V2); d1(V2); d2(V1).
3 ~  :- d1(V2); d2(V1); not d6(V1).
3 ~  :- d0(V1); d6(V1); not d1(V1).
3 ~  :- d0(V2); d2(V1); d6(V1).
3 ~ d2(V1) :- d0(V1); not d1(V1).
3 ~  :- d0(V1); d1(V1); d2(V1).
3 ~  :- d0(V1); d2(V1); not d1(V1).
3 ~  :- d0(V2); d2(V1); not d6(V1).
3 ~  :- d0(V1); not d1(V1); not d6(V1).
3 ~  :- d0(V2); d2(V1); not d1(V1).
3 ~  :- d0(V3); d1(V2); d6(V1).
3 ~  :- d2(V1); d6(V1); not d1(V1).
3 ~  :- d0(V2); d2(V1); not d6(V2).
3 ~ d2(V1) :- d6(V1); not d1(V1).
3 ~  :- d2(V1); not d1(V1); not d6(V1).
3 ~  :- d2(V2); d6(V1); not d1(V1).
3 ~  :- d0(V2); d1(V1); d2(V1).
3 ~  :- d0(V2); d1(V2); d6(V1).
3 ~  :- d0(V2); d6(V1); not d2(V1).
3 ~  :- d1(V2); d2(V1); not d6(V2).
3 ~  :- d0(V2); d6(V1); not d1(V1).
3 ~  :- d0(V1); d1(V1); d6(V1).
3 ~ d2(V1) :- d0(V2); d6(V1).
3 ~  :- d0(V2); d1(V1); d6(V1).
3 ~  :- d2(V2); d6(V1); not d1(V2).
3 ~  :- d0(V2); d1(V1); not d6(V1).
3 ~  :- d2(V2); d6(V1); not d0(V2).
3 ~ d2(V1) :- d0(V1); d6(V1).
3 ~  :- d0(V2); d1(V1); not d6(V2).
3 ~  :- d1(V2); d6(V1); not d0(V2).
3 ~  :- d1(V2); d2(V1); not d0(V2).
3 ~ d2(V1) :- d0(V2); d1(V1).
3 ~  :- d2(V2); d6(V1); not d0(V1).
3 ~  :- d1(V2); d2(V1); not d0(V1).
3 ~ d2(V1) :- d6(V1); not d0(V1).
3 ~  :- d1(V1); d2(V1); not d0(V1).
3 ~  :- d2(V1); not d0(V1); not d1(V1).
3 ~ d2(V1) :- d1(V1); not d0(V1).
3 ~ d2(V1) :- d1(V2); d6(V1).
3 ~  :- d6(V1); not d0(V1); not d1(V1).
3 ~  :- d0(V1); d1(V2); d6(V1).
3 ~  :- d1(V1); d6(V1); not d0(V1).
3 ~  :- d1(V1); not d0(V1); not d6(V1).
3 ~  :- d2(V1); not d0(V1); not d6(V1).
3 ~  :- d1(V2); d6(V1); not d0(V1).
3 ~  :- d2(V1); d6(V1); not d0(V1).
4 ~ d2(V1) :- d0(V1); d1(V2); d6(V1).
4 ~ d2(V1) :- d1(V2); d6(V1); not d0(V1).
4 ~ d2(V1) :- d1(V1); not d0(V1); not d6(V1).
4 ~ d2(V1) :- d6(V1); not d0(V1); not d1(V1).
4 ~ d2(V1) :- d1(V1); d6(V1); not d0(V1).
4 ~ d2(V1) :- d1(V1); d2(V2); not d0(V1).
4 ~ d2(V1) :- d2(V2); d6(V1); not d0(V1).
4 ~ d2(V1) :- d1(V1); d6(V2); not d0(V2).
4 ~ d2(V1) :- d1(V1); d6(V2); not d0(V1).
4 ~ d2(V1) :- d1(V1); d2(V2); not d0(V2).
4 ~ d2(V1) :- d0(V2); d1(V1); not d6(V1).
4 ~ d2(V1) :- d0(V2); d1(V1); not d6(V2).
4 ~ d2(V1) :- d1(V2); d6(V1); not d0(V2).
4 ~ d2(V1) :- d2(V2); d6(V1); not d0(V2).
4 ~ d2(V1) :- d0(V1); d6(V2); not d1(V2).
4 ~ d2(V1) :- d0(V2); d1(V1); d6(V1).
4 ~ d2(V1) :- d2(V2); d6(V1); not d1(V2).
4 ~ d2(V1) :- d0(V1); d2(V2); not d1(V2).
4 ~ d2(V1) :- d0(V2); d6(V1); not d1(V1).
4 ~ d2(V1) :- d1(V1); d2(V2); not d6(V2).
4 ~ d2(V1) :- d0(V1); d2(V2); not d6(V2).
4 ~ d2(V1) :- d0(V2); d1(V2); d6(V1).
4 ~ d2(V1) :- d0(V1); d6(V2); not d1(V1).
4 ~ d2(V1) :- d2(V2); d6(V1); not d1(V1).
4 ~ d2(V1) :- d0(V3); d1(V2); d6(V1).
4 ~ d2(V1) :- d0(V1); d1(V1); d6(V1).
4 ~ d2(V1) :- d0(V1); d2(V2); not d1(V1).
4 ~ d2(V1) :- d0(V1); not d1(V1); not d6(V1).
4 ~ d2(V1) :- d0(V1); d6(V1); not d1(V1).
4 ~ d2(V1) :- d0(V1); d1(V2); not d6(V1).
4 ~ d2(V1) :- d0(V2); d1(V1); d2(V2).
4 ~ d2(V1) :- d0(V2); d2(V2); d6(V1).
4 ~ d2(V1) :- d1(V1); d2(V2); not d6(V1).
4 ~ d2(V1) :- d0(V2); d6(V1); not d1(V2).
4 ~ d2(V1) :- d0(V1); d1(V1); not d6(V1).
4 ~ d2(V1) :- d0(V1); d2(V2); not d6(V1).
4 ~ d2(V1) :- d0(V2); d1(V1); d6(V2).
4 ~ d2(V1) :- d1(V1); d6(V2); not d2(V2).
4 ~ d2(V1) :- d0(V1); d6(V2); not d2(V2).
4 ~ d2(V1) :- d0(V1); d1(V2); not d2(V2).
4 ~ d2(V1) :- d1(V2); d6(V1); not d2(V2).
4 ~ d2(V1) :- d1(V1); d2(V3); d6(V2).
4 ~ d2(V1) :- d0(V2); d1(V1); not d2(V2).
4 ~ d2(V1) :- d0(V1); d2(V3); d6(V2).
4 ~ d2(V1) :- d0(V2); d6(V1); not d2(V2).
4 ~ d2(V1) :- d1(V1); d2(V2); d6(V2).
4 ~ d2(V1) :- d0(V1); d2(V2); d6(V2).
4 ~ d2(V1) :- d0(V3); d1(V1); d2(V2).
4 ~ d2(V1) :- d0(V3); d2(V2); d6(V1).
4 ~ d2(V1) :- d1(V2); d2(V2); d6(V1).
4 ~ d2(V1) :- d1(V3); d2(V2); d6(V1).
4 ~ d2(V1) :- d0(V1); d2(V2); d6(V1).
4 ~ d2(V1) :- d1(V1); d2(V2); d6(V1).
4 ~ d2(V1) :- d0(V1); d1(V1); d2(V2).
4 ~ d2(V1) :- d0(V1); d1(V2); d2(V2).
4 ~ d2(V1) :- d0(V1); d1(V3); d2(V2).
4 ~ d2(V1) :- d0(V3); d1(V1); d6(V2).
4 ~ d2(V1) :- d0(V1); d1(V3); d6(V2).
4 ~ d2(V1) :- d0(V1); d1(V2); d6(V2).
4 ~ d2(V1) :- d0(V1); d1(V1); d6(V2).
4 ~ d2(V1) :- d0(V1); d1(V2); not d6(V2).
