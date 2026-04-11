d4(o0).
d0(o1).
d2(o3).
d4(o5).
d0(o3).

d1(X) :- not d2(X), d4(X).
d5(X) :- d2(X).
d4(X) :- d2(X).
d0(X) :- d4(X).