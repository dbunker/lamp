d4(o4).
d0(o1).
d2(o1).
d4(o2).
d0(o4).

d1(X) :- d4(X), not d2(X).
d1(X) :- d0(X), not d2(X).