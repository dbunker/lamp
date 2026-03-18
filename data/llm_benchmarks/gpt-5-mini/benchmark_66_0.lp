d0(o0).
d2(o0).
d5(o4).
d0(o4).
d5(o1).
d1(o2).
d2(o1).
d5(o2).
d4(o3).

d0(X) :- d5(X).
d4(X) :- d2(X), not d1(X).
d2(X) :- d4(X).
d2(X) :- d5(X).