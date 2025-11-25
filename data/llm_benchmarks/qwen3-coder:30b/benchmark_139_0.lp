d1(o4).
d4(o3).
d1(o3).
d1(o1).
d0(o4).

d4(X) :- not d0(X), d1(X).
d3(X) :- not d4(X), d2(X).
d2(X) :- not d0(X), d1(X).