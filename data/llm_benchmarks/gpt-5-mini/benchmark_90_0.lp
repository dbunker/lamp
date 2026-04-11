d2(o2).
d3(o2).
d4(o2).
d5(o1).
d3(o5).
d3(o4).
d2(o0).
d2(o1).
d0(o4).

d0(X) :- not d2(X), d4(X).
d1(X) :- not d4(X), d2(X).
d4(X) :- not d2(X), d3(X).
d3(X) :- not d5(X), d2(X).