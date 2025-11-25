d3(o0).
d2(o2).
d1(o3).
d2(o1).

d1(X) :- not d0(X), d2(X).
d0(X) :- not d2(X), d1(X).