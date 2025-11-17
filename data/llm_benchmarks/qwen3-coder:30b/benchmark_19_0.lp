d2(o3).
d2(o2).
d1(o3).
d4(o1).

d0(X) :- not d2(X), d4(X).
d0(X) :- not d2(X), d1(X).